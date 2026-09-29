# tqp-dht: a BitTorrent DHT source for the TurboQuant Pro console
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""A BitTorrent DHT source for the TurboQuant Pro console.

``tqp_dht.daemon`` runs one libtorrent session that seeds official open-source
images and serves its state on 127.0.0.1; ``tqp_dht.krpc`` reads the DHT's
packets and rebuilds each lookup's convergence. ``tqp console --dht URL`` draws
the snapshot.
"""

__version__ = "0.1.0"
