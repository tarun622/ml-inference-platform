"""Closed-loop async load generator for quick, repeatable latency/throughput numbers.

Uses raw keep-alive HTTP/1.1 connections with TCP_NODELAY and one write per
request. Generic Python HTTP clients send headers and body in separate writes,
which interacts with Nagle's algorithm and delayed ACKs and adds ~40 ms per
request on Linux, measuring the client rather than the server.

  python loadtest/bench.py --url http://localhost:8080 --concurrency 1 16 64 --requests 20000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Conn:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def open(self) -> None:
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        self.writer.get_extra_info("socket").setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    async def post(self, path: str, body: bytes) -> int:
        if self.writer is None:
            await self.open()
        head = (
            f"POST {path} HTTP/1.1\r\nHost: {self.host}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: keep-alive\r\n\r\n"
        ).encode()
        self.writer.write(head + body)
        await self.writer.drain()
        status_line = await self.reader.readline()
        if not status_line:
            raise ConnectionError("server closed connection")
        status = int(status_line.split()[1])
        length = 0
        while (line := await self.reader.readline()) not in (b"\r\n", b""):
            k, _, v = line.decode().partition(":")
            if k.lower() == "content-length":
                length = int(v)
            elif k.lower() == "connection" and v.strip().lower() == "close":
                self.writer = None
        await self.reader.readexactly(length)
        return status

    async def close(self) -> None:
        if self.writer:
            self.writer.close()


async def run(url: str, conc: int, n: int, model: str, batch: int) -> dict:
    u = urlparse(url)
    rows = np.load(ROOT / "model_repository" / model / "testset.npz")["X"][:1000].tolist()
    path = f"/v1/models/{model}/predict"
    bodies = [json.dumps({"instances": random.sample(rows, batch)}).encode() for _ in range(500)]
    lat: list[float] = []
    errors = 0
    remaining = n

    async def worker():
        nonlocal errors, remaining
        c = Conn(u.hostname, u.port or 80)
        for _ in range(20):  # warm-up
            await c.post(path, bodies[0])
        while remaining > 0:
            remaining -= 1
            t = time.perf_counter()
            try:
                if await c.post(path, random.choice(bodies)) != 200:
                    errors += 1
            except (OSError, ConnectionError, asyncio.IncompleteReadError):
                errors += 1
                c = Conn(u.hostname, u.port or 80)
            lat.append(time.perf_counter() - t)
        await c.close()

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(conc)))
    elapsed = time.perf_counter() - t0
    lat.sort()

    def q(p: float) -> float:
        return round(1000 * lat[min(len(lat) - 1, int(p * len(lat)))], 2)

    return {
        "concurrency": conc,
        "batch": batch,
        "requests": len(lat),
        "rps": round(len(lat) / elapsed),
        "p50_ms": q(0.5),
        "p95_ms": q(0.95),
        "p99_ms": q(0.99),
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--requests", type=int, default=10000)
    ap.add_argument("--model", default="fraud_mlp")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    results = []
    for c in args.concurrency:
        results.append(asyncio.run(run(args.url, c, args.requests, args.model, args.batch)))
        print(json.dumps(results[-1]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
