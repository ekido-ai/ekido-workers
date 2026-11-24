# Q8 export: risks before implementation

Related to [internal #135](https://github.com/ekido-ai/ekido-internal/issues/135).
The signed network manifest format and coordinator consumption need agreement;
this work prepares local export metadata only.

1. A checkpoint from a different job or a modified checkpoint could be exported.
   Bind the existing checkpoint lineage and record its canonical manifest hash.
2. A failed conversion or a changing/truncated output could leave an apparently
   usable file. Record an artifact only after conversion completes, check its
   GGUF magic, and hash the bytes without loading the whole model into RAM.
3. A second request or a device/process failure during export could leave stale
   or partially written metadata. Do not publish success until an atomic metadata
   update succeeds; network integration still needs an exclusive job lease.
4. A path escaping the private job directory could expose another customer's
   file. Reject artifact paths outside the export directory.
5. A malicious provider can misstate quantization or the model's quality. The
   receiver must verify artifact bytes, signature, assignment and quality verdict;
   a local hash alone is not an authorization or quality decision.

The reviewer should add another failure mode. Network encryption, signing and
end-to-end lifecycle checks are not provided by this local building block.
