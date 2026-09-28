# One Trillion Vectors, One Recall Number

TurboQuant Pro just finished its largest run. One trillion 4-bit compressed vectors, sharded across 500 servers, 24 terabytes of index on block storage, built and measured end to end on the National Research Platform. Then we did the expensive thing nobody does at this size. We scanned the entire index exactly and asked what the routed search had missed.

𝗧𝗵𝗲 𝗻𝘂𝗺𝗯𝗲𝗿. Routed inverted-file search, probing 128 of 2048 cells, returns 0.999 of the exact scan's top ten at a trillion rows. At 32 probes, 0.989. The same protocol gave 0.999 at a hundred million rows, at ten billion, and at a hundred billion. Four orders of magnitude, one number. Routing over shards that share a rotation and a coarse quantizer loses nothing the exact scan finds, and the global top ten is an exact merge of the per-shard top tens, so this is a measurement, not an estimate.

𝗪𝗵𝗮𝘁 𝗶𝘁 𝗺𝗲𝗮𝗻𝘀. This is the routing layer's number, taken against the compressed index's own exact scan, on a seed-defined synthetic corpus. It is not recall against the truth, and this project has a rule about that. At a billion rows, where we kept the original float vectors, the compressed scan found 0.592 of the true neighbours and reranking the shortlist against the originals brought it to 0.991. On 15 million real Cohere embeddings the same pattern holds. The codes find, the originals decide. Trust the ground truth, not the self-consistency check, and never accept on reconstruction cosine.

𝗛𝗼𝘄 𝗶𝘁 𝗿𝗮𝗻. Every job in the run, 500 exact scans and 500 routed passes, fit in one CPU and two gigabytes of memory. That is the smallest resource class on a shared research cluster, and it is the right one for a fleet of independent, idempotent jobs. Three things make a two-billion-row scan fit there. Read the query set from a cache instead of deriving it. Hold two memory-mapped shards open at a time, since even a mapped shard carries its row ids in RAM. Score in blocks of 65 thousand rows. The whole measurement cost 772 CPU-hours.

The submissions went through the same polite-submit path the earlier fleet runs used. Twenty jobs in flight, paced by a controller that yields when the cluster is busy, every job resumable from its own output on the volume. Over two days that pool carried 928 submissions to 849 completions with no operator on duty, across the ordinary weather of a shared cluster, nodes that go away mid-job and pods that wait a long time to start. A job whose output already exists prints that fact and exits, so a retry costs a minute and nothing else.

𝗪𝗵𝗮𝘁 𝗶𝘀 𝗶𝗻 𝘁𝗵𝗲 𝗿𝗲𝗽𝗼. Everything. The seeds that define the corpus, so any worker regenerates any row bit for bit. The driver and its state files. The pilots, the interim score taken two thirds of the way through, and the final score with every per-server wall time. If you want to argue with the number, the record is there to argue with.

Thanks to the National Research Platform and its operators for hosting the run.

pip install turboquant-pro
https://github.com/ahb-sjsu/turboquant-pro

The PVLDB writeup is in progress.
