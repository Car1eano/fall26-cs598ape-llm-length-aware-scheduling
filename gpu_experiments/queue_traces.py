"""Local OTLP receiver for vLLM's per-request queue measurements."""

import json
import re
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Condition, Thread

REQUEST_ID = re.compile(r"^cmpl-cs598-measure-(\d+)-0$")


@contextmanager
def queue_receiver(path, expected_ids):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest, ExportTraceServiceResponse,
    )

    collected = {}
    condition = Condition()

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            message = ExportTraceServiceRequest.FromString(payload)
            with condition:
                for resource in message.resource_spans:
                    for scope in resource.scope_spans:
                        for span in scope.spans:
                            if span.name != "llm_request":
                                continue
                            attrs = {item.key: item.value for item in span.attributes}
                            match = REQUEST_ID.fullmatch(attrs["gen_ai.request.id"].string_value)
                            if match:
                                request_id = int(match.group(1))
                                collected[request_id] = attrs["gen_ai.latency.time_in_queue"].double_value
                condition.notify_all()
            result = ExportTraceServiceResponse().SerializeToString()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        def log_message(self, *_):
            pass

    with HTTPServer(("127.0.0.1", 0), Receiver) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def finish():
            with condition:
                complete = condition.wait_for(lambda: collected.keys() == expected_ids, timeout=15)
                path.write_text("".join(json.dumps({"id": key, "queue_time_s": value}) + "\n"
                                        for key, value in sorted(collected.items())))
                if not complete:
                    raise RuntimeError(f"Only {len(collected)}/{len(expected_ids)} queue traces captured")

        try:
            yield f"http://127.0.0.1:{server.server_port}/v1/traces", finish
        finally:
            server.shutdown()
            thread.join()
