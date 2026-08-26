Skills are short markdown playbooks for procedures that recur across nodes.

A role sees only the *index* (filename + first line) in its system prompt, and reads
the body with `read_file` when it decides the skill applies — so adding skills costs
almost no context. Any agent may add one; the meta agent is expected to promote a
procedure here once it has worked twice.

Keep each under ~80 lines. Name them by the task, e.g. `encode-clips-to-latents.md`.
