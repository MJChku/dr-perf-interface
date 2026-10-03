"""Check the actual LMCache EXISTS coroutine with a held socket response.

No GPU, timing estimate, replacement EXISTS implementation, or model required.
The socket pair replaces the transport endpoint, not the connector method.
"""
import asyncio
import inspect
import json
import socket
import threading

import torch
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import ClientMetaMessage, ServerMetaMessage, ServerReturnCode
from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector


def check():
    client, server = socket.socketpair()
    loop = asyncio.new_event_loop()
    worker = threading.Thread(target=loop.run_forever, daemon=True)
    worker.start()
    # EXISTS only uses these two fields. Preserve the real connector method.
    connector = object.__new__(LMCServerConnector)
    connector.client_socket = client
    connector.async_socket_lock = asyncio.Lock()
    key = CacheEngineKey("check", 1, 0, 123, torch.float16)
    received, release, canary = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def reply():
        try:
            server.settimeout(5)
            request = b""
            while len(request) < ClientMetaMessage.packlength():
                part = server.recv(ClientMetaMessage.packlength() - len(request))
                if not part:
                    raise RuntimeError("EOF in request")
                request += part
            received.set()
            assert release.wait(5), "response release timed out"
            server.sendall(ServerMetaMessage(ServerReturnCode.SUCCESS, 0,
                MemoryFormat(1), torch.float16, torch.Size([0, 0, 0, 0])).serialize())
        except BaseException as exc:
            errors.append(repr(exc))
        finally:
            server.close()

    peer = threading.Thread(target=reply, daemon=True)
    peer.start()
    future = asyncio.run_coroutine_threadsafe(connector.exists(key), loop)
    try:
        assert received.wait(5), "EXISTS request did not reach peer"
        loop.call_soon_threadsafe(canary.set)
        ran_while_response_held = canary.wait(0.2)
        release.set()
        result = future.result(5)
        ran_after_release = canary.wait(5)
        assert not ran_while_response_held and ran_after_release and result is True
        peer.join(5)
        assert not errors, errors
        return dict(method=inspect.getsourcefile(LMCServerConnector.exists),
            responseHeldMs=200, unrelatedCallbackRanWhileHeld=ran_while_response_held,
            unrelatedCallbackRanAfterRelease=ran_after_release, existsResult=result,
            conclusion="A blocking EXISTS response stalls the connector event loop.")
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        worker.join(5)
        client.close()
        if not worker.is_alive():
            loop.close()


print(json.dumps([check() for _ in range(3)], indent=2))
