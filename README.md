# Stickman Shorts Factory

Makes 9:16 YouTube Shorts automatically on GitHub Actions, with free tools only:
cartoon stills from a free key-less image AI, Kokoro narration, camera moves, transitions,
word-by-word captions and a synthesized music bed.

- A job is a folder `jobs/<YYYY-MM-DD>/` with one JSON spec per Short (see the existing jobs).
- Pushing a job to `main` starts the workflow; finished videos are pushed to the branch `out-<YYYY-MM-DD>`.
- The daily Claude scheduled task writes the jobs, collects the videos and sends them to Himanshu.
