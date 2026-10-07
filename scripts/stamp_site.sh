#!/usr/bin/env bash
# Writes the current commit + UTC date into the site-build line of docs/index.html docs/discovery-chain.html docs/series-deck.html. Run before commit.
set -e
cd "$(dirname "$0")/.."
sha=$(git rev-parse --short HEAD); d=$(date -u +%Y-%m-%d\ %H:%MZ)
perl -0pi -e "s|Site build <code>[^<]*</code>|Site build <code>after $sha · $d</code>|" docs/index.html docs/discovery-chain.html docs/series-deck.html
