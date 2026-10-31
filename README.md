## Data

This project uses two public datasets. Neither dataset is stored in this repository. Download them yourself and place them in `data/raw/` (see below).

### Steam
- **Steam Games Dataset 2025** (Ermilov et al., 2025), Kaggle: https://www.kaggle.com/datasets/artermiloff/steam-games-dataset
- Licence: MIT. Keep the dataset's copyright notice from the Kaggle page with any copy you hold.
- Use: training, validation and testing of the Steam pipeline.

### Google Play
- **GamesDB public dataset**, file `dataset_playstore1.csv`, Hugging Face: https://huggingface.co/datasets/GamesDB/gamesdb_public_dataset
- Licence: Public Domain Dedication and License (PDDL), as stated on the repository card.
- Content: a June 2021 scrape of 2,312,944 apps with 24 columns. It holds store-listing metadata only and no description text.
- Use: training, validation and testing of the Google Play pipeline. Games are selected from 14 game genres, and a game is labelled well received if its average rating is 4.0 or higher with at least 50 ratings.
- Privacy: the developer email column is dropped when the file is loaded and is never stored or used. Developers are kept only as hashed identifiers for grouped cross-validation.

### Notes
- The Google Play file was scraped from a public store by the dataset owner. It is used here for private, non-commercial academic research and is not redistributed.
- The Steam and Google Play labels, populations and snapshot dates differ, so results for the two platforms are reported side by side and are not compared as like-for-like.
- The Google Play models (tiers M1 to M3) are planned. Only preliminary single-model results exist so far, and all reported figures are preliminary.