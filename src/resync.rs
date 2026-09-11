//! Tier-2 resync targets: which `?resync=` URLs the addon will fetch, and the relay that feeds the
//! chosen stream to alass.
//!
//! `?resync=` is a URL picked by whoever sends the request, and the addon fetches it server-side for
//! up to 90 seconds. Unguarded, that is a blind GET from inside the box to anything the box can
//! reach — every LAN host, and `den-embed:8080` on the container network — plus attacker-chosen
//! media fed to ffmpeg's parsers. Two rules close it:
//!
//!   * **Only den-scout's play routes at an operator-listed origin.** A target must be
//!     `<origin>/<config>/play/<token>` or `<origin>/p/<ticket>`, with `<origin>` in `SCOUT_ORIGINS`.
//!     Those are the URLs the Den app sends (the playing source's scout link: the legacy form
//!     carries the install config, the ticket form a short-lived play ticket), and scout answers
//!     them only for a valid install config or ticket — so a caller without one cannot make scout
//!     redirect anywhere. With `SCOUT_ORIGINS`
//!     unset, Tier 2 is off. A "public addresses only" fallback was the obvious alternative and is
//!     not safe: an attacker's public server can answer with an HLS playlist, and ffmpeg opens every
//!     segment URL in it itself — `http://192.168.x.y/…` included — without this process ever
//!     seeing the request.
//!
//!   * **ffmpeg never touches the network.** alass hands its input straight to `ffprobe` and
//!     `ffmpeg -i`, which resolve the name again (a DNS-rebinding window after any check made here)
//!     and follow redirects on their own, and alass takes no ffmpeg options to stop either. So alass
//!     is given a relay on 127.0.0.1 instead of the upstream URL. The relay follows the redirect
//!     chain itself — scout's `/play` is a 302 to the debrid CDN, so redirects cannot simply be
//!     refused — resolving each hop once and refusing any hop that is neither a listed origin nor a
//!     public address. It then pins the vetted address for every range request ffmpeg makes, and
//!     answers any later redirect with a 502, so there is nothing for ffmpeg to follow.

use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::combinators::UnsyncBoxBody;
use http_body_util::{BodyExt, Empty, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, LOCATION, RANGE};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use reqwest::Url;
use tokio::net::TcpListener;
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::timeout;

/// Redirects followed from scout's `/play` to the bytes. Scout answers with one 302 to the debrid
/// CDN; a CDN may add one or two of its own.
const MAX_HOPS: usize = 5;

/// One hop of the redirect chain: scout resolving the debrid link, or the CDN's first byte.
const HOP_BUDGET: Duration = Duration::from_secs(15);

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Bytes one resync may pull through the relay. Tier 2's 90-second budget is the tighter bound on
/// any real link; this one holds even if that budget ever grows.
const MAX_RELAY_BYTES: u64 = 8 << 30;

/// An origin allowed to be a resync target: scheme, lower-cased host, and explicit port (the
/// scheme's default filled in, so `http://h` and `http://h:80` are the same origin).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Origin {
    scheme: String,
    host: String,
    port: u16,
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}://{}:{}", self.scheme, self.host, self.port)
    }
}

/// `SCOUT_ORIGINS`: comma-separated origins such as `http://192.168.86.193:8080`. An entry that is not
/// a bare http(s) origin — a path, a query, credentials — is ignored with a warning rather than
/// widened into one that would match more than the operator wrote.
pub fn parse_origins(raw: &str) -> Vec<Origin> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|s| {
            let origin = parse_origin(s);
            if origin.is_none() {
                eprintln!("warning: SCOUT_ORIGINS entry {s:?} is not an http(s) origin — ignored");
            }
            origin
        })
        .collect()
}

fn parse_origin(s: &str) -> Option<Origin> {
    Url::parse(s)
        .ok()
        .filter(|u| u.path() == "/" && u.query().is_none() && u.fragment().is_none())
        .and_then(|u| origin_of(&u))
}

/// `SCOUT_ALIASES`: comma-separated `<public origin>=<LAN origin>` pairs, such as
/// `https://d-play.oxy.fi=http://192.168.86.193:8080`. A resync target on a public name is fetched at
/// its LAN address instead: scout is on this box, and two services on one box must not need the WAN,
/// Cloudflare and the tunnel to reach each other (oxyc/den#15). Both sides still have to be in
/// `SCOUT_ORIGINS` — the public one for `vet`, the LAN one for the hop it becomes.
pub fn parse_aliases(raw: &str) -> Vec<(Origin, Origin)> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            let parsed = pair
                .split_once('=')
                .and_then(|(p, l)| Some((parse_origin(p.trim())?, parse_origin(l.trim())?)));
            if parsed.is_none() {
                eprintln!(
                    "warning: SCOUT_ALIASES entry {pair:?} is not <public origin>=<LAN origin> — ignored"
                );
            }
            parsed
        })
        .collect()
}

/// `url` at its LAN address when its origin is a public name in `aliases`; anything else as it is.
pub fn local(mut url: Url, aliases: &[(Origin, Origin)]) -> Url {
    let Some(origin) = origin_of(&url) else { return url };
    if let Some((_, lan)) = aliases.iter().find(|(public, _)| *public == origin) {
        // Scheme first: setting a port the old scheme treats as its default would otherwise be dropped.
        let _ = url.set_scheme(&lan.scheme);
        let _ = url.set_host(Some(&lan.host));
        let _ = url.set_port(Some(lan.port));
    }
    url
}

/// The origin of a plain http(s) URL. A URL carrying credentials has none: `http://ok-host@evil/` is
/// a request to `evil`, and nothing here should ever read it otherwise.
fn origin_of(url: &Url) -> Option<Origin> {
    if !matches!(url.scheme(), "http" | "https") || !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    Some(Origin {
        scheme: url.scheme().to_string(),
        host: url.host_str()?.to_ascii_lowercase(),
        port: url.port_or_known_default()?,
    })
}

/// The longest `/p/<ticket>` segment accepted. A ticket seals every debrid account on the install
/// (so scout's multi-account logic still works), and the largest config scout admits comes to about
/// 5.9 KB; scout caps a path segment at 8 KiB, so this matches that cap rather than cutting under it.
const MAX_TICKET_LEN: usize = 8192;

/// The resync target, normalized, if it is one of den-scout's play routes at a listed origin —
/// `/<config>/play/<token>`, or `/p/<ticket>` — and `None` for anything else, and for everything
/// when no origin is listed.
///
/// No DNS here. A listed origin is trusted by name because the operator named it; where the
/// redirect chain after it leads is `Relay::open`'s question, asked once per hop.
pub fn vet(raw: &str, origins: &[Origin]) -> Option<Url> {
    let url = Url::parse(raw).ok()?;
    if !origins.contains(&origin_of(&url)?) || url.query().is_some() || url.fragment().is_some() {
        return None;
    }
    let segments: Vec<&str> = url.path_segments()?.collect();
    match segments.as_slice() {
        [config, "play", token] if !config.is_empty() && !token.is_empty() => Some(url),
        ["p", ticket] if is_ticket(ticket) => Some(url),
        _ => None,
    }
}

/// A play ticket as scout mints it: one base64url segment, unpadded. Opaque here — whether it is
/// valid is scout's call — so this only holds the shape exact: nothing percent-encoded, no padding,
/// nothing that could turn into a second segment.
fn is_ticket(s: &str) -> bool {
    (1..=MAX_TICKET_LEN).contains(&s.len())
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// An address on the public internet: not loopback, private (RFC 1918), CGNAT (100.64/10, where a
/// tailnet lives), link-local (the 169.254.169.254 metadata endpoint), IPv6 ULA, multicast,
/// documentation, benchmarking or reserved space — and not an IPv6 form that carries one of those
/// IPv4 addresses (mapped, NAT64, 6to4).
pub fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_public_v4(v4);
            }
            let s = v6.segments();
            let embedded = |hi: u16, lo: u16| {
                let [a, b] = hi.to_be_bytes();
                let [c, d] = lo.to_be_bytes();
                Ipv4Addr::new(a, b, c, d)
            };
            // NAT64 (64:ff9b::/96) and 6to4 (2002::/16) reach the IPv4 address they carry.
            if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
                return is_public_v4(embedded(s[6], s[7]));
            }
            if s[0] == 0x2002 {
                return is_public_v4(embedded(s[1], s[2]));
            }
            !(s[..6] == [0; 6] // ::/96: unspecified, loopback, the deprecated IPv4-compatible block
                || v6.is_multicast()
                || s[0] & 0xfe00 == 0xfc00 // ULA fc00::/7
                || s[0] & 0xffc0 == 0xfe80 // link-local fe80::/10
                || s[0] & 0xffc0 == 0xfec0 // site-local fec0::/10
                || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation
                || (s[0] == 0x2001 && s[1] == 0)) // Teredo
        }
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || a == 0
        || (a == 100 && b & 0xc0 == 64) // CGNAT 100.64/10
        || (a == 192 && b == 0 && c == 0) // IETF protocol assignments
        || (a == 198 && b & 0xfe == 18) // benchmarking 198.18/15
        || a >= 240) // reserved
}

/// A client for one hop, pinned to the one address vetted for it.
///
/// Pinned because a name resolved once for the check and again for the connection can answer
/// differently the second time. No redirects, because each one is a new hop that has to be vetted
/// the same way. No proxy, because a proxy from the environment would decide the destination
/// instead of the address pinned here.
async fn pinned_client(url: &Url, origins: &[Origin]) -> Result<reqwest::Client, String> {
    let origin = origin_of(url).ok_or("resync hop is not a plain http(s) URL")?;
    let listed = origins.contains(&origin);
    // `host_str` brackets an IPv6 literal.
    let bare = origin.host.trim_start_matches('[').trim_end_matches(']');
    let builder = reqwest::Client::builder()
        .user_agent(concat!("den-subtitles/", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .connect_timeout(CONNECT_TIMEOUT);
    let builder = match bare.parse::<IpAddr>() {
        Ok(ip) if listed || is_public_ip(ip) => builder,
        Ok(_) => return Err("resync hop is not a public address".into()),
        Err(_) => {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((bare, origin.port))
                .await
                .map_err(|e| format!("resolve resync hop: {e}"))?
                .collect();
            // Every address, not just the one pinned: a name that also answers with an internal
            // address is not one to be trusted with the rest.
            if !listed && addrs.iter().any(|a| !is_public_ip(a.ip())) {
                return Err("resync hop resolves to a non-public address".into());
            }
            let first = *addrs.first().ok_or("resync hop resolves to nothing")?;
            builder.resolve(bare, first)
        }
    };
    builder.build().map_err(|e| format!("resync client: {e}"))
}

/// Follow `start`'s redirects to the URL that serves the bytes, vetting each hop, and return it with
/// the client pinned to its address. Error text never carries a URL: a debrid link is a credential.
async fn follow(start: Url, origins: &[Origin]) -> Result<(reqwest::Client, Url), String> {
    let mut url = start;
    for _ in 0..=MAX_HOPS {
        let client = pinned_client(&url, origins).await?;
        // One byte: this request only asks where the bytes are, and the relay's requests fetch them.
        let resp = timeout(HOP_BUDGET, client.get(url.clone()).header(RANGE, "bytes=0-0").send())
            .await
            .map_err(|_| "resync target timed out".to_string())?
            .map_err(|e| format!("resync target: {}", e.without_url()))?;
        let status = resp.status();
        if status.is_redirection() {
            let location = resp
                .headers()
                .get(LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or("resync redirect without a location")?;
            url = url.join(location).map_err(|_| "resync redirect to an unparseable location")?;
            continue;
        }
        if status.is_success() {
            return Ok((client, url));
        }
        return Err(format!("resync target answered {status}"));
    }
    Err("resync target redirected too many times".into())
}

type RelayBody = UnsyncBoxBody<Bytes, std::io::Error>;

struct Upstream {
    client: reqwest::Client,
    target: Url,
    budget: AtomicU64,
}

/// A loopback HTTP server serving one resync target to alass's ffprobe and ffmpeg for as long as the
/// handle lives. Dropping it stops the server and every connection on it.
pub struct Relay {
    addr: SocketAddr,
    task: JoinHandle<()>,
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Relay {
    /// Resolve `target` (already passed through `vet`) to the bytes and start relaying them. A target on
    /// a public name in `aliases` starts at its LAN address; the hops after it are the debrid's.
    pub async fn open(
        target: &str,
        origins: &[Origin],
        aliases: &[(Origin, Origin)],
    ) -> Result<Relay, String> {
        let start = Url::parse(target).map_err(|_| "resync target is not a URL".to_string())?;
        let start = local(start, aliases);
        let (client, target) = follow(start, origins).await?;
        let listener =
            TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.map_err(|e| format!("relay bind: {e}"))?;
        let addr = listener.local_addr().map_err(|e| format!("relay bind: {e}"))?;
        let upstream = Arc::new(Upstream { client, target, budget: AtomicU64::new(MAX_RELAY_BYTES) });
        Ok(Relay { addr, task: tokio::spawn(accept(listener, upstream)) })
    }

    /// What to hand alass in place of the upstream URL.
    pub fn url(&self) -> String {
        format!("http://{}/media", self.addr)
    }
}

async fn accept(listener: TcpListener, upstream: Arc<Upstream>) {
    // Owned by this task, so aborting it drops the set, which aborts every connection with it.
    let mut conns = JoinSet::new();
    let http = hyper::server::conn::http1::Builder::new();
    while let Ok((stream, _)) = listener.accept().await {
        while conns.try_join_next().is_some() {}
        let upstream = upstream.clone();
        let service = service_fn(move |req| {
            let upstream = upstream.clone();
            async move { Ok::<_, Infallible>(relay(upstream, req).await) }
        });
        let conn = http.serve_connection(TokioIo::new(stream), service);
        conns.spawn(async move {
            let _ = conn.await;
        });
    }
}

/// Forward one request to the pinned target. Only the `Range` header goes up, and only the headers
/// a demuxer needs to seek come back. Any answer but content is a 502: a 3xx passed through is a
/// redirect ffmpeg would follow on its own.
async fn relay(up: Arc<Upstream>, req: Request<Incoming>) -> Response<RelayBody> {
    let request = match *req.method() {
        Method::GET => up.client.get(up.target.clone()),
        Method::HEAD => up.client.head(up.target.clone()),
        _ => return bare(StatusCode::METHOD_NOT_ALLOWED),
    };
    let request = match req.headers().get(RANGE) {
        Some(range) => request.header(RANGE, range.clone()),
        None => request,
    };
    let Ok(resp) = request.send().await else { return bare(StatusCode::BAD_GATEWAY) };
    let status = resp.status();
    if !matches!(status, StatusCode::OK | StatusCode::PARTIAL_CONTENT | StatusCode::RANGE_NOT_SATISFIABLE) {
        return bare(StatusCode::BAD_GATEWAY);
    }
    let mut out = Response::builder().status(status);
    for name in [CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, ACCEPT_RANGES] {
        if let Some(v) = resp.headers().get(&name) {
            out = out.header(name, v.clone());
        }
    }
    let body = resp.bytes_stream().map(move |chunk| {
        let chunk = chunk.map_err(|e| std::io::Error::other(e.without_url()))?;
        match spend(&up.budget, chunk.len() as u64) {
            true => Ok(Frame::data(chunk)),
            false => Err(std::io::Error::other("resync relay byte budget spent")),
        }
    });
    out.body(StreamBody::new(body).boxed_unsync()).unwrap_or_else(|_| bare(StatusCode::BAD_GATEWAY))
}

/// Take `n` bytes from the relay's budget, or refuse without taking any.
fn spend(budget: &AtomicU64, n: u64) -> bool {
    budget.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| left.checked_sub(n)).is_ok()
}

fn bare(status: StatusCode) -> Response<RelayBody> {
    let mut resp =
        Response::new(Empty::<Bytes>::new().map_err(|never: Infallible| match never {}).boxed_unsync());
    *resp.status_mut() = status;
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn origins(list: &str) -> Vec<Origin> {
        parse_origins(list)
    }

    #[test]
    fn origins_are_normalized_and_bad_entries_dropped() {
        let got = origins(" http://192.168.86.193:8080 , HTTPS://Scout.Example.com/, http://h:80,  ");
        assert_eq!(
            got.iter().map(ToString::to_string).collect::<Vec<_>>(),
            ["http://192.168.86.193:8080", "https://scout.example.com:443", "http://h:80"]
        );
        // Anything that would match more, or other, than a bare origin is ignored.
        for bad in ["http://h/sub", "http://h/?q=1", "http://u@h", "ftp://h", "h:8080", "not a url"] {
            assert!(origins(bad).is_empty(), "accepted {bad:?}");
        }
    }

    #[test]
    fn only_scout_play_urls_at_a_listed_origin_pass() {
        let list = origins("http://192.168.86.193:8080,https://scout.example.com");
        let ok = [
            "http://192.168.86.193:8080/eyJjZmci/play/abc123",
            "https://scout.example.com/cfg/play/tok",
            "https://scout.example.com:443/cfg/play/tok", // the default port, written out
            "https://SCOUT.example.com/cfg/play/tok",     // hosts compare case-insensitively
            // The ticket form: one base64url segment under /p/.
            "http://192.168.86.193:8080/p/AbC-_09xyz",
            "https://scout.example.com/p/tkt",
        ];
        for u in ok {
            assert!(vet(u, &list).is_some(), "refused {u}");
        }
        let refused = [
            // Wrong route shape at a listed origin.
            "http://192.168.86.193:8080/cfg/stream/movie/tt1.json",
            "http://192.168.86.193:8080/cfg/play/",
            "http://192.168.86.193:8080//play/tok",
            "http://192.168.86.193:8080/cfg/play/tok/extra",
            "http://192.168.86.193:8080/play/tok",
            "http://192.168.86.193:8080/health",
            "http://192.168.86.193:8080/cfg/play/tok?probe=1",
            "http://192.168.86.193:8080/cfg/play/tok#x",
            // The right shape anywhere else: another port, scheme, host, or any private address.
            "http://192.168.86.193:8081/cfg/play/tok",
            "https://192.168.86.193:8080/cfg/play/tok",
            "http://192.168.86.194:8080/cfg/play/tok",
            "http://10.0.0.1/cfg/play/tok",
            "http://den-embed:8080/cfg/play/tok",
            "http://127.0.0.1:8080/cfg/play/tok",
            "http://localhost:8080/cfg/play/tok",
            "http://169.254.169.254/cfg/play/tok",
            "http://[::1]:8080/cfg/play/tok",
            "http://[fd00::1]:8080/cfg/play/tok",
            "http://[::ffff:192.168.86.193]:8080/cfg/play/tok",
            // A name that merely starts with, or embeds, a listed address.
            "http://192.168.86.193.attacker.tld:8080/cfg/play/tok",
            "http://10.0.0.1.attacker.tld/cfg/play/tok",
            // Credentials: the real host is after the `@`, and credentials are refused outright.
            "http://192.168.86.193:8080@evil.tld/cfg/play/tok",
            "http://user@192.168.86.193:8080/cfg/play/tok",
            // Not http(s), or not a URL.
            "file:///etc/passwd",
            "ftp://192.168.86.193:8080/cfg/play/tok",
            "",
            // The ticket form in any shape but exactly /p/<one base64url segment>.
            "http://192.168.86.193:8080/p",
            "http://192.168.86.193:8080/p/",
            "http://192.168.86.193:8080/p/tkt/extra",
            "http://192.168.86.193:8080/p/tkt?probe=1",
            "http://192.168.86.193:8080/p/tkt#x",
            "http://192.168.86.193:8080/p/tk%2Ft", // percent-encoded
            "http://192.168.86.193:8080/p/tkt=",   // padded
            "http://192.168.86.193:8080/p/tk.t",
            "http://192.168.86.193:8080/p/tk+t",
            "http://192.168.86.193:8080/P/tkt",
            "http://192.168.86.193:8080/pp/tkt",
            "http://192.168.86.193:8080/cfg/p/tkt",
            "http://192.168.86.193:8080//p/tkt",
            // The ticket form anywhere else, or with credentials.
            "http://192.168.86.193:8081/p/tkt",
            "http://10.0.0.1/p/tkt",
            "http://127.0.0.1:8080/p/tkt",
            "http://user@192.168.86.193:8080/p/tkt",
            "http://192.168.86.193:8080@evil.tld/p/tkt",
        ];
        for u in refused {
            assert!(vet(u, &list).is_none(), "accepted {u}");
        }
    }

    #[test]
    fn a_ticket_is_bounded() {
        let list = origins("http://192.168.86.193:8080");
        let ticket = |n: usize| format!("http://192.168.86.193:8080/p/{}", "A".repeat(n));
        assert!(vet(&ticket(MAX_TICKET_LEN), &list).is_some());
        assert!(vet(&ticket(MAX_TICKET_LEN + 1), &list).is_none());
    }

    #[test]
    fn with_no_origin_listed_nothing_passes() {
        assert!(vet("http://192.168.86.193:8080/cfg/play/tok", &[]).is_none());
        assert!(vet("https://scout.example.com/cfg/play/tok", &[]).is_none());
        assert!(vet("http://192.168.86.193:8080/p/tkt", &[]).is_none());
    }

    #[test]
    fn only_public_addresses_are_public() {
        let public = ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808"];
        for ip in public {
            assert!(is_public_ip(ip.parse().unwrap()), "{ip} counted as internal");
        }
        let internal = [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.3.4",
            "192.168.86.193",
            "100.64.0.1",      // CGNAT / tailnet
            "100.127.255.255", // CGNAT's last address
            "169.254.169.254", // cloud metadata
            "0.0.0.0",
            "0.1.2.3",
            "255.255.255.255",
            "224.0.0.1",
            "192.0.0.1",
            "192.0.2.1",
            "198.18.0.1",
            "240.0.0.1",
            "::",
            "::1",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "fec0::1",
            "ff02::1",
            "2001:db8::1",
            "2001::1",            // Teredo
            "::ffff:127.0.0.1",   // IPv4-mapped loopback
            "::ffff:10.0.0.1",    // IPv4-mapped private
            "::127.0.0.1",        // IPv4-compatible
            "64:ff9b::7f00:1",    // NAT64 of loopback
            "2002:c0a8:5601::1",  // 6to4 of 192.168.86.1
            "64:ff9b::a9fe:a9fe", // NAT64 of the metadata endpoint
        ];
        for ip in internal {
            assert!(!is_public_ip(ip.parse().unwrap()), "{ip} counted as public");
        }
        // 100.128/10 is past CGNAT: ordinary public space.
        assert!(is_public_ip("100.128.0.1".parse().unwrap()));
    }

    #[test]
    fn the_byte_budget_refuses_without_spending() {
        let budget = AtomicU64::new(10);
        assert!(spend(&budget, 6));
        assert!(!spend(&budget, 5), "overspent the budget");
        assert_eq!(budget.load(Ordering::Relaxed), 4, "a refused chunk still took bytes");
        assert!(spend(&budget, 4));
        assert!(!spend(&budget, 1));
    }

    /// What one test server answers: status, headers, body.
    type Answer = (u16, Vec<(&'static str, String)>, Bytes);

    /// A loopback HTTP server answering every request with `handler`, counting the requests it gets.
    async fn server<F>(handler: F) -> (SocketAddr, Arc<AtomicUsize>)
    where
        F: Fn(&Request<Incoming>, usize) -> Answer + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let (handler, counter) = (Arc::new(handler), hits.clone());
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (handler, counter) = (handler.clone(), counter.clone());
                let service = service_fn(move |req: Request<Incoming>| {
                    let n = counter.fetch_add(1, Ordering::SeqCst);
                    let (status, headers, body) = handler(&req, n);
                    let mut resp = Response::new(http_body_util::Full::new(body));
                    *resp.status_mut() = StatusCode::from_u16(status).unwrap();
                    for (k, v) in headers {
                        resp.headers_mut().insert(k, v.parse().unwrap());
                    }
                    async move { Ok::<_, Infallible>(resp) }
                });
                tokio::spawn(
                    hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), service),
                );
            }
        });
        (addr, hits)
    }

    fn redirect(to: String) -> Answer {
        (302, vec![("location", to)], Bytes::new())
    }

    /// Serves `media` with range support, the way a debrid CDN does.
    fn ranged(req: &Request<Incoming>, media: &'static [u8]) -> Answer {
        let range =
            req.headers().get(RANGE).and_then(|v| v.to_str().ok()).and_then(|r| r.strip_prefix("bytes="));
        let Some((a, b)) = range.and_then(|r| r.split_once('-')) else {
            return (200, vec![("accept-ranges", "bytes".into())], Bytes::from_static(media));
        };
        let a: usize = a.parse().unwrap();
        let b: usize = b.parse::<usize>().unwrap_or(media.len() - 1).min(media.len() - 1);
        let range = format!("bytes {a}-{b}/{}", media.len());
        (206, vec![("content-range", range)], Bytes::from_static(&media[a..=b]))
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none()).build().unwrap()
    }

    /// The bypass the old guard had: a listed scout origin (or any public URL) that 302s to loopback
    /// or the LAN. ffmpeg followed it; the relay refuses the hop before connecting.
    #[tokio::test]
    async fn a_redirect_to_an_internal_address_is_not_followed() {
        let (victim, victim_hits) = server(|_, _| (200, vec![], Bytes::from_static(b"secret"))).await;
        let port = victim.port();
        for location in [
            format!("http://127.0.0.1:{port}/latest/meta-data/"),
            format!("http://localhost:{port}/"),
            format!("http://[::1]:{port}/"),
            format!("http://[::ffff:127.0.0.1]:{port}/"),
            format!("http://0x7f.1:{port}/"), // a spelling of 127.0.0.1 the URL parser normalizes
            "http://169.254.169.254/latest/meta-data/".to_string(),
            "file:///etc/passwd".to_string(),
        ] {
            let (scout, _) = server(move |_, _| redirect(location.clone())).await;
            let list = origins(&format!("http://{scout}"));
            let target = vet(&format!("http://{scout}/cfg/play/tok"), &list).unwrap();
            let err = Relay::open(target.as_str(), &list, &[]).await.err();
            assert!(err.is_some(), "followed a redirect to an internal address");
        }
        assert_eq!(victim_hits.load(Ordering::SeqCst), 0, "the internal address was contacted");
    }

    /// A legitimate resync still works end to end: scout's 302 is followed and ffmpeg gets working
    /// range requests through the relay. (The hop stays on the listed origin — tests have no public
    /// address to redirect to.)
    #[tokio::test]
    async fn the_relay_serves_ranges_from_the_followed_target() {
        const MEDIA: &[u8] = b"0123456789";
        let (scout, _) = server(|req, _| match req.uri().path() {
            "/cfg/play/tok" => redirect("/cdn/film.mkv".into()),
            _ => ranged(req, MEDIA),
        })
        .await;
        let list = origins(&format!("http://{scout}"));
        let target = vet(&format!("http://{scout}/cfg/play/tok"), &list).unwrap();
        let relay = Relay::open(target.as_str(), &list, &[]).await.expect("a legitimate target was refused");
        assert!(relay.url().starts_with("http://127.0.0.1:"), "{}", relay.url());

        let resp = client().get(relay.url()).header(RANGE, "bytes=2-5").send().await.unwrap();
        assert_eq!(resp.status(), 206);
        assert_eq!(resp.headers()[CONTENT_RANGE], "bytes 2-5/10");
        assert_eq!(resp.bytes().await.unwrap(), &b"2345"[..]);

        let whole = client().get(relay.url()).send().await.unwrap();
        assert_eq!(whole.status(), 200);
        assert_eq!(whole.bytes().await.unwrap(), MEDIA);

        let addr = relay.url();
        drop(relay);
        tokio::task::yield_now().await;
        assert!(client().get(addr).send().await.is_err(), "the relay outlived its handle");
    }

    /// A ticket URL is followed through scout's redirect exactly like the legacy play route.
    #[tokio::test]
    async fn the_relay_follows_a_ticket_url() {
        const MEDIA: &[u8] = b"ticketed";
        let (scout, _) = server(|req, _| match req.uri().path() {
            "/p/AbC-_09" => redirect("/cdn/film.mkv".into()),
            _ => ranged(req, MEDIA),
        })
        .await;
        let list = origins(&format!("http://{scout}"));
        let target = vet(&format!("http://{scout}/p/AbC-_09"), &list).expect("a ticket URL was refused");
        let relay = Relay::open(target.as_str(), &list, &[]).await.expect("a ticket target was not relayed");
        let whole = client().get(relay.url()).send().await.unwrap();
        assert_eq!(whole.bytes().await.unwrap(), MEDIA);
    }

    /// A ticket on a public name (`SCOUT_ALIASES`) is fetched at scout's LAN address: the public name
    /// would only lead back to this box through the tunnel, so it is never contacted.
    #[tokio::test]
    async fn a_public_ticket_url_is_fetched_at_its_lan_address() {
        const MEDIA: &[u8] = b"aliased";
        let (scout, hits) = server(|req, _| match req.uri().path() {
            "/p/AbC-_09" => redirect("/cdn/film.mkv".into()),
            _ => ranged(req, MEDIA),
        })
        .await;
        let lan = format!("http://{scout}");
        let list = origins(&format!("https://d-play.invalid,{lan}"));
        let aliases = parse_aliases(&format!("https://d-play.invalid={lan}, not-a-pair"));
        assert_eq!(aliases.len(), 1);
        let target =
            vet("https://d-play.invalid/p/AbC-_09", &list).expect("the public ticket URL was refused");
        let relay =
            Relay::open(target.as_str(), &list, &aliases).await.expect("the aliased ticket was not relayed");
        assert_eq!(client().get(relay.url()).send().await.unwrap().bytes().await.unwrap(), MEDIA);
        assert!(hits.load(Ordering::SeqCst) >= 2, "scout's LAN address served the ticket and the bytes");
        let moved = local(Url::parse("https://d-play.invalid/p/x?q=1").unwrap(), &aliases);
        assert_eq!(moved.as_str(), format!("{lan}/p/x?q=1"));
        let kept = local(Url::parse("https://cdn.example/f.mkv").unwrap(), &aliases);
        assert_eq!(kept.as_str(), "https://cdn.example/f.mkv");
    }

    /// The target can change its answer after it was vetted. A redirect then reaches the relay, not
    /// ffmpeg, and goes no further.
    #[tokio::test]
    async fn a_later_redirect_is_a_502_and_is_not_passed_on() {
        let (victim, victim_hits) = server(|_, _| (200, vec![], Bytes::from_static(b"secret"))).await;
        let (scout, _) = server(move |_, n| match n {
            0 => (206, vec![("content-range", "bytes 0-0/1".into())], Bytes::from_static(b"x")),
            _ => redirect(format!("http://{victim}/")),
        })
        .await;
        let list = origins(&format!("http://{scout}"));
        let target = vet(&format!("http://{scout}/cfg/play/tok"), &list).unwrap();
        let relay = Relay::open(target.as_str(), &list, &[]).await.unwrap();

        let resp = client().get(relay.url()).send().await.unwrap();
        assert_eq!(resp.status(), 502);
        assert!(resp.headers().get(LOCATION).is_none(), "a redirect reached the relay's client");
        assert_eq!(victim_hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn an_endless_redirect_chain_is_cut_off() {
        let (scout, hits) = server(|_, n| redirect(format!("/cfg/play/tok{n}"))).await;
        let list = origins(&format!("http://{scout}"));
        let target = vet(&format!("http://{scout}/cfg/play/tok"), &list).unwrap();
        assert!(Relay::open(target.as_str(), &list, &[]).await.is_err());
        assert_eq!(hits.load(Ordering::SeqCst), MAX_HOPS + 1);
    }
}
