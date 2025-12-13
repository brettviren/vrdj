The vrdj (poorly) tries to do what real human DJs do: find songs that go well together. It does this by identifying a set of songs that are similar to a seed song or a set of seed songs. Similarity is based on applying the VGGish embedding model to form vectors and then finding vectors for other songs that are need the vector(s) for the seed(s).

The vrdj works best when used as a beets plugin.


# Install

Here is my current "kitchen sink" install of beets including vrdj

```
$ uv tool install \
        --with git+https://github.com/brettviren/vrdj \
        --with git+https://github.com/igordertigor/beets-usertag \
        beets[discogs,chroma,autobpm]
```


# Beets plugin


## Configuration

```yaml
plugins: vrdj
vrdj:
  auto: no
  device: cpu
  metric: cosine
  embedding: vggish
```

This shows the defaults.

-   **auto:** If yes, embeddings and vectors for songs are inserted into the vrdj database during import time. Calculating embeddings takes 1-10 seconds depending on CPU/GPU.
-   **device:** In principle, VGGish can run on GPU to somewhat accelerate generating the embedding. Expect something like 1-10s per song depending on length and hardware. Embeddings are cached in vrdj's DB and not regenerated.
-   **metric:** The comparison metric. **cosine** is probably best and compares vector directions but you can try **l2** which will be sensitive to vector length which encodes loudness.
-   **embedding:** Currently only VGGish is supported but in the future maybe another is added. VGGish is trained on all sorts of sounds and so music of all kinds tends to cluster together. Expect all songs to have cosine similarity of 0.9 or higher.


## Usage

```
$ beet vrdj --help
$ beet vrdj <seed-query> [options]
```

Any seeds found by the query that are not yet ingested will have their VGGish embedding calculated and a vector added to the FAISS index. This information is kept in files that by default reside next to the Beets DB file location.

It supports Beets path variable formatted printing:

```
$ beet vrdj <seed-query> --format '$path'
```

You can override that to get out an M3U playlist

```
$ beet vrdj <seed-query> --playlist > playlist.m3u
```

By default, up to 10 similar items are emitted. You can change that with the `-n|--number` option.


# Others in this space

-   Beets' chroma / acoustic ID. This can be used to evaluate "equality" if not "distance"
-   [vibenet](https://github.com/jaeheonshim/vibenet) gives a 7-dimensional vector in a "music emotion" semantically meaningful space (energy, speechiness, etc). It could be one day added to vrdj as an embedding. Or, perhaps find a way to usefully combine vibenet and vggish embeddings.


# Some design decisions

vrdj uses its own sqlite DB file plus one or more FAISS index files. It does not modify the Beets DB file. The vrdj DB holds the full VGGish embedings and a mapping between a song's embedding, the Beets item id and the id for each of possibly many FAISS indices. The motivation for this:

1.  FAISS indices need to be their own file in any case.
2.  A mapping between Beets and FAISS needs to go somewhere.
    -   FAISS index can be pre song or per time segment per song.
    -   See options below.
3.  The VGGish embedding is kind of large and opaque and if it was put into Beets DB it would be easy and, I think, unwanted to `beet write` it to file tags.

Given the use of FAISS already means managing new files in addition to the Beets sqlite file, adding yet another file for a central vrdj sqlite file doesn't seem so bad and it allows for future extension.


## Discarded options

Actually, FAISS has a `IndexIDMap` which gives us a 64bit int to associate with each index entry. This could be used to directly hold a packing of the Beets index and the segment number (or zero for whole-song vectors). If VGGish embeddings were not kept then there would be no need for a vrdj db. OTOH, this case would make it hard to know that a particular song was FAISS-indexed.


## Design issues needing work

Removing items from Beets DB is not reflected in vrdj DB or the FAISS indices. When FAISS includes these missing Beets entries vrdj will return fewer that requested songs. Nuking the FAISS index file is a reasonable option as it is fairly quick to regenerate. The bloat of the VGGish embeddings will be retained in the vrdj DB.
