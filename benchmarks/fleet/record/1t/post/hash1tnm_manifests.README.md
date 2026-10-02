# Checksum manifests of the 10^12-row index

There is one manifest for each of the 500 servers, `hash1tnm_part_0.json` to `hash1tnm_part_499.json`. Each one gives the sha256 and byte count of every file on that server's index volume (1,201 files per server). The ref jobs of run `1tnm` wrote them (`fleet_ref.write_hash`, see `driver1tnm_hash.log`).

The manifests total 73 MB, so they live outside git, as one asset on a data release:

- Release: https://github.com/ahb-sjsu/turboquant-pro/releases/tag/data-1t-hash-manifests-2026-10-02
- Asset: `hash1tnm_manifests.tar.gz`, 26,260,907 bytes
- sha256: `cea55f0e369b427abeecd86c1c181aef49b79f7c1bfe01b36e75461598333303`

To check a download against this record:

    sha256sum hash1tnm_manifests.tar.gz      # must equal the sha256 above
    tar xzf hash1tnm_manifests.tar.gz
    sha256sum -c hash1tnm_manifests.SHA256SUMS   # the file beside this one: 500 lines, all OK

The tarball is deterministic: names sorted, mtime 2026-10-02 00:00 UTC, owner 0, gzip -n -9.

The index was complete on all 500 servers on 2026-09-24, and its volumes were released on 2026-10-02. A rebuild from the corpus seeds and the code in `benchmarks/fleet/` can be checked file by file against these manifests.
