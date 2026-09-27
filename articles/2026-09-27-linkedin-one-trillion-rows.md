# We Searched a Trillion Vectors and the Recall Number Didn't Move

This weekend a run I have been nursing since early August finished. TurboQuant Pro built and measured an index of one trillion 4-bit compressed vectors, 500 servers of two billion rows each, 24 terabytes on disk, on a shared academic Kubernetes cluster with no reserved hardware. Then it scanned the whole thing exactly, to see what the routed search had missed.

The answer is nothing it could see.

𝗧𝗵𝗲 𝗻𝘂𝗺𝗯𝗲𝗿. Routed inverted-file search, probing 128 of 2048 cells, returned 0.999 of the exact scan's top ten. At 32 probes, 0.989. The same protocol gave 0.999 at a hundred million rows, at ten billion, and at a hundred billion. Four orders of magnitude, one number.

𝗪𝗵𝗮𝘁 𝗶𝘁 𝗺𝗲𝗮𝗻𝘀, 𝗮𝗻𝗱 𝘄𝗵𝗮𝘁 𝗶𝘁 𝗱𝗼𝗲𝘀𝗻'𝘁. The index is sharded, and every shard shares one rotation and one coarse quantizer, so scores are comparable across shards and the global top ten is an exact merge of per-shard top tens. The result says that when the corpus grows by adding shards, the routing layer loses nothing the exact scan finds. That is the claim.

Here is the part I insist on saying out loud, because I have watched this kind of number get repeated without it. This is recall against the compressed index's own exact scan, on a synthetic corpus whose intrinsic dimension is about 16. It is not recall against the truth. At a billion rows, where we kept the original float vectors, the compressed scan found only 0.592 of the true neighbours, and reranking the shortlist against the originals brought that to 0.991. On 15 million real Cohere embeddings the same pattern holds. The routing transfers. The compression at four bits does not, on its own. Trust the tail, not the mean, and trust the ground truth, not the self-consistency check.

𝗧𝗵𝗲 𝗽𝗮𝗿𝘁 𝗜 𝗳𝗼𝘂𝗻𝗱 𝗺𝗼𝗿𝗲 𝗶𝗻𝘁𝗲𝗿𝗲𝘀𝘁𝗶𝗻𝗴. The cluster this ran on enforces resource usage. Ask for six CPUs and use half of one, and an enforcer deletes your pod. The only workload it never touches is one CPU and two gigabytes. So every job in this run had to fit in two gigabytes, including a full scan of two billion rows per server.

Three pilots died teaching me how:

1️⃣ The scan regenerated its 100 queries from four five-million-row blocks at startup. 640 MB each. Dead in 85 seconds. Read the cached file instead.

2️⃣ A memory-mapped shard still builds 45 MB of row ids in RAM, and the library kept 128 shards open by default. 5.8 GB over a scan. Dead at minute 17. Keep two open.

3️⃣ Score in blocks of 65 thousand rows so the temporaries stay small.

After that, the scan ran at 88 percent of one CPU and 61 percent of two gigabytes, and the whole measurement, 500 exact scans and 1000 routed passes, cost 772 CPU-hours. Not a single job was ever killed by the enforcer.

𝗧𝗵𝗲 𝗼𝘁𝗵𝗲𝗿 𝗵𝗮𝗹𝗳 𝗼𝗳 𝘁𝗵𝗲 𝘄𝗼𝗿𝗸 𝘄𝗮𝘀 𝗻𝗼𝘁 𝘁𝗵𝗲 𝗶𝗻𝗱𝗲𝘅. It was a pool driver that submits twenty jobs at a time to a shared cluster and copes. 928 submissions for 849 completions over two days. One host killed every job placed on it with a bus error. Three nodes went unreachable mid-scan. Twenty-six pods sat pending for 45 minutes and were re-issued, and in every case the original had actually run, so the retry found its output in minutes. Every job is idempotent, so none of this cost anything but time.

The bug that cost the most was mine. The submission controller politely defers job creation when the namespace already has five pending, and my driver treated a deferred job as a vanished one and resubmitted it, queueing a duplicate behind every original. When another group's GPU jobs started pending at 3 AM, throughput fell from 20 servers an hour to 3. The fix is one number. Wait ninety minutes before you decide something is lost.

𝗪𝗵𝗮𝘁 𝗶𝘀 𝗶𝗻 𝘁𝗵𝗲 𝗿𝗲𝗽𝗼. Everything. The seeds that define the corpus, so any worker regenerates any row bit for bit. The driver and its state files. Both driver logs, the one from the attempt that got stopped and the one that finished. The three pilots that died and the one that passed. The interim score I took two thirds of the way through, and the final score with every per-server wall time. If you want to argue with the number, the record is there to argue with.

Thanks to the National Research Platform and its operators, who let a two-day run in the exempt class share their cluster, and gave a thumbs up when I announced it.

pip install turboquant-pro
https://github.com/ahb-sjsu/turboquant-pro

The PVLDB writeup is in progress. The one-line version is already in the title.
