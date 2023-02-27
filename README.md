# Abstract

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/C2DH/template_repo_JDH/main?filepath=author_guideline_template.ipynb)

The cultural reuse and reappropriation of audiovisual content has been a recurring topic of research in the humanities, not least in studies of remix cultures. An open question that remains, however, is how artificial intelligence and machine learning may help scholars study the reuse and reappropriation of audiovisual heritage. In this article, we introduce the Video Reuse Detector (VRD) – a methodological toolkit for identifying visual similarities in audiovisual archives with the help of machine learning. Designed to assist in the study of the “social life” and “cultural biographies” of video clips, the VRD helps explore how the meaning of historic footage changes when it circulates and is recycled/cross-referenced in video productions through time. The toolkit uses machine learning techniques (specifically, convolutional neural networks), combined with tools for performing similarity searches (specifically, the Faiss library) to detect copies and near-copies in audiovisual archives. It also assembles a series of tools for trimming and preparing datasets and filtering/visualizing matching results (such as introducing similarity thresholds, filtering based on sequential matches of frames, and visually viewing the final matching results). Inspired by the “visual turn” in digital history and digital humanities research, the article will introduce the basic logic and rationale behind the VRD, exemplify how the toolkit works, and discuss how the digitization of audiovisual archives open for new ways of exploring the reuse of historic moving images.


# Keywords
Digital methods, Machine learning, Cultural reuse, Video archives


# Licenses
In this demo, the VRD is applied to two videos that are openly available on Archive.org:

[The Eagle Has Landed: The Flight of Apollo 11](https://archive.org/details/gov.archives.arc.45017) (28 min) released by the U.S. National Aeronautics and Space Administration (NASA) in 1969. Licensed under CC0 1.0 Universal.

[The Moon and Us](https://archive.org/details/journey-through-the-solar-system-episode-06-the-moon-us) (28 min) Episode 6, in the documentary series Journey Through the Solar System released by NASA in 1983. Licensed under Attribution-NonCommercial-NoDerivs 4.0 International.

