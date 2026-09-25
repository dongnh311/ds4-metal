# Ornith 23G ICE: llama.cpp oracle self-check

- Model: `Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`, 22,836,518,208 bytes,
  from gbuzhf/Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF.
- llama.cpp: version: 0.5.0 (build 11146, commit 7fe450e19), Homebrew, Metal (`-ngl 99 -fa on`).
- Corpus: repo `corpora/code.test.raw`, 64 chunks, n_ctx 2048.
- Published: PPL(BF16) 2.194208, 23G PPL ratio 1.0018 -> 2.1982.
- Measured: Final estimate: PPL = 2.1985 +/- 0.01559.
- Verdict: PASS.
