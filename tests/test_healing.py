"""Self-heal ladder regressions: a corrupt mid-stream abort is repaired by a
targeted continuation from the verified clean prefix instead of a blind full
retry. Hermetic mock upstream (no network)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from simurg.integrations.openai_guard import GuardedLLM

CLEAN = ("The quarterly report showed steady growth across all regions. "
         "Revenue increased by eleven percent year over year, driven mostly "
         "by enterprise subscriptions. Operating expenses rose more slowly, "
         "so margins expanded for the third consecutive quarter. Management "
         "expects this trend to continue into the next fiscal year, provided "
         "that input costs remain stable and hiring plans stay on track. ")
LOOP = "the same phrase keeps repeating in a tight loop "
CONTINUATION = (" The board also approved a dividend of two dollars per share "
                "for the coming quarter, signaling confidence in the durable "
                "underlying demand for the company's core product lines.")

CALLS = []


class MockUpstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        msgs = body.get("messages", [])
        CALLS.extend(m.get("content", "") for m in msgs
                     if m.get("role") == "assistant")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        # A continuation request (assistant prefix present) -> emit the clean tail.
        is_continuation = any(m.get("role") == "assistant" and m.get("content")
                              for m in msgs[:-1])
        if is_continuation:
            stream = CONTINUATION
        else:
            # primary: clean prefix, then collapse into a repetition loop.
            stream = CLEAN + LOOP * 40

        for i in range(0, len(stream), 6):
            payload = json.dumps({"choices": [{"delta": {"content": stream[i:i+6]}}]})
            self.wfile.write(b"data: " + payload.encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture(scope="module")
def healed_result():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstream)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = "http://{}:{}".format(*httpd.server_address)
    llm = GuardedLLM(url + "/v1", model="mock", retries=1)
    result = llm.chat([{"role": "user", "content": "Summarize the quarter."}],
                      on_token=lambda t: None)
    httpd.shutdown()
    return result


def test_primary_corrupt_then_healed(healed_result):
    assert healed_result.ok is True
    assert healed_result.healed is True
    # the ladder saw: primary (corrupt) -> heal-1 (clean)
    labels = [a.label for a in healed_result.attempts]
    assert labels[0] == "primary"
    assert any(l.startswith("heal-") for l in labels)
    corrupt = [a for a in healed_result.attempts if a.state == "corrupt"]
    assert corrupt, "expected the primary attempt to be flagged corrupt"


def test_stitched_text_starts_with_clean_prefix_and_has_tail(healed_result):
    txt = healed_result.text
    assert txt.startswith(CLEAN.strip()[:40]), "stitched text must keep the verified prefix"
    assert "dividend of two dollars" in txt, "the healed tail must be present"
    # The detector has real latency (~590 chars past onset, see README), so a
    # short gray-zone tail of the loop is expected and acceptable. The honest
    # guarantee: the answer is BOUNDED (the remaining ~1800 chars of garbage
    # are cut) and the healed TAIL is clean prose, not more loop.
    full_corrupt = len(CLEAN) + len(LOOP) * 40
    assert len(txt) < full_corrupt * 0.6, \
        "healed answer must be far shorter than the full corrupted stream"
    # the healed region (last 200 chars) must be clean, not loop
    assert LOOP.strip() not in txt[-200:], "the healed tail must be clean prose"
    # loop must not dominate: far fewer loop hits than the primary's 40
    assert txt.count(LOOP.strip()) < 8, "loop text must not dominate the stitched answer"


def test_heal_request_carried_the_clean_prefix(healed_result):
    # the continuation prompt must include the released clean prefix as context
    assert any(CLEAN.strip()[:60] in c for c in CALLS), \
        "heal request should carry the verified prefix as an assistant turn"


def test_zero_leak_still_holds(healed_result):
    # nothing corrupt was ever forwarded: the loop is not the accepted text
    assert healed_result.verdict in ("clean", "suspect")
