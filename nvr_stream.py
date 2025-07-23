import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime

def get_nvr_token(host, username, password, channel, start_time, overlap=100):
    """
    Obtain a session token by performing Digest authentication against the security.cgi endpoint.
    Uses the RTSP playback URI for the specified channel and lets requests handle Digest handshake.
    """
    url = f"http://{host}/stw-cgi/security.cgi"
    rtsp_uri = f"rtsp://{host}/PlaybackChannel/{channel}/media.smp/session=0"
    params = {
        "msubmenu": "digestauth",
        "action": "view",
        "Uri": rtsp_uri,
        "start": start_time,
        "overlap": str(overlap)
    }
    headers = {"Accept": "application/json"}

    # Automatic Digest Auth handshake
    resp = requests.get(
        url,
        auth=HTTPDigestAuth(username, password),
        params=params,
        headers=headers
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("Response")
    if not token or token.lower() == "fail":
        raise RuntimeError(f"Failed to obtain token: {data}")
    return token


def stream_mjpeg(host, username, password, channel, start_time, overlap, token, out_file=None):
    """
    Fetch the MJPEG stream using the given token, optionally writing to a file.
    """
    url = f"http://{host}/uwa/media/ump/Worker/MjpegSession"
    params = {
        "CameraUID": str(channel),
        "StartTime": start_time,
        "Overlap": str(overlap),
        "Response": token
    }
    headers = {"Accept": "*/*"}

    with requests.get(
        url,
        auth=HTTPDigestAuth(username, password),
        params=params,
        headers=headers,
        stream=True
    ) as r:
        r.raise_for_status()
        if out_file:
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=4096):
                    if chunk:
                        f.write(chunk)
        else:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    print(chunk)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='NVR Digest auth + MJPEG stream')
    parser.add_argument('--host', default='192.168.1.18', help='NVR IP or hostname')
    parser.add_argument('--user', default='admin', help='Username')
    parser.add_argument('--password', required=True, help='Password')
    parser.add_argument('--channel', default=0, type=int, help='Camera channel number')
    parser.add_argument(
        '--start',
        default=datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
        help='Start time (UTC) in YYYYMMDDThhmmssZ format'
    )
    parser.add_argument('--overlap', default=100, type=int, help='Frame overlap in ms')
    parser.add_argument('--out', help='Output file to save MJPEG stream')
    args = parser.parse_args()

    token = get_nvr_token(
        args.host, args.user, args.password,
        args.channel, args.start, args.overlap
    )
    print(f"Obtained token: {token}")

    stream_mjpeg(
        args.host,
        args.user,
        args.password,
        args.channel,
        args.start,
        args.overlap,
        token,
        out_file=args.out
    )
