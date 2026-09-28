# Measure It or Say Why

TurboQuant Pro has a console now. One screen in the terminal, in the spirit of btop, that shows what the library is doing while it does it. It runs over SSH, never opens a browser, and fits the whole picture on one grid.

𝗪𝗵𝗮𝘁 𝗶𝘁 𝘀𝗵𝗼𝘄𝘀. Nine panels. System load. Throughput and latency. The query pipeline, stage by stage, in milliseconds. An oscilloscope and a spectrum analyzer for any signal the library emits. ReadScope, which shows the observer a result was certified for, the certificate, and whether that certificate still holds on the data in front of it. The index. A NATS fabric panel. And a live query stream where any query opens into its own trace and can be replayed. Every search path traces itself, from the flat compressed scan through IVF, sharded, HNSW, FAISS, adaptive rerank and scatter-gather, one trace per call.

𝗧𝗵𝗲 𝗿𝘂𝗹𝗲. A panel with no real data measures it or says why. Every number carries its unit, its source and its kind, measured, sampled, derived or estimated, and the kind is printed next to it. Missing is shown as missing, with the reason, never as a zero. A certificate has four states, not two. Valid, stale, inconclusive or unchecked, computed live against the originals. Nothing on the screen is decoration, and nothing is green by default.

𝗪𝗵𝗮𝘁 𝘁𝗵𝗲 𝘀𝘂𝗿𝘃𝗲𝘆 𝗳𝗼𝘂𝗻𝗱. Before drawing a single panel, we surveyed every feature of the library, one row each across retrieval, models and evidence, and asked of each one what a panel would actually be able to read. That survey paid for itself before any code changed. It showed that only one search path was instrumented, so a console pointed at a real index would have sat quietly while it ran. And it measured the Hugging Face KV cache drop-in spilling one token at a time during decode, where the keys cost more than seven times their fp16 size. Spilling in blocks brought them to about a third of fp16, and that fix shipped first. A dashboard would have drawn a pleasant picture over both. Instruments that can only agree with you cannot help you.

𝗛𝗼𝘄 𝗶𝘁 𝗶𝘀 𝗯𝘂𝗶𝗹𝘁. Two processes. A Python engine does the measuring on one BLAS thread at low priority. A small Go client, standard library only, owns the terminal. They talk over a private Unix socket, and the client never waits on the engine, so a slow measurement never freezes the screen. The client treats the terminal the way btop does. It restores your shell before every exit and every suspend, and a test suite drives real interactive shells through each path to prove it. On our lab server the client uses about 1% of one core and the engine 37% while serving the demo at 20 queries a second.

𝗧𝗵𝗲 𝗸𝗲𝘆𝘀. Tab moves between panels and the focused panel takes the keys. On the scope, 1 to 4 turn channels on and off, and Shift with the number selects one for scale and autoset. z zooms a panel to the full screen. P writes a text snapshot you can paste into an issue.

pip install turboquant-pro
tqp console --demo
https://github.com/ahb-sjsu/turboquant-pro

The Go client builds once from go/tqp-console, and the console prints the command if it is missing. Bundling it into the wheel is next.

𝗜𝗳 𝘆𝗼𝘂 𝗰𝗮𝗻𝗻𝗼𝘁 𝘀𝗲𝗲 𝗶𝘁, 𝘆𝗼𝘂 𝗰𝗮𝗻𝗻𝗼𝘁 𝗰𝗲𝗿𝘁𝗶𝗳𝘆 𝗶𝘁.

#vectorsearch #observability #quantization #RAG #opensource #terminal #golang #python
