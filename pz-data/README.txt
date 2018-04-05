======= TERMS OF USAGE ======= 

The Parzival database may be used for non-commercial research and teaching purposes only. If you are publishing scientific work based on the Parzival database, we request you to include a reference to:

A. Fischer, A. Keller, V. Frinken, and H. Bunke: "Lexicon-Free Handwritten Word Spotting Using Character HMMs," in Pattern Recognition Letters, Volume 33(7), pages 934-942, 2012.

======= DATA SET =======

* IDs

The word ID "d-006a-001_01" can be read as follows: manuscript d (Abbey Library of Saint Gall, Switzerland, Cod. 857), page 6, column a, line 1, word 1. The line ID is "d-006a-001" and the page ID is "d-006" accordingly. Word and line numbers start at 1.

* sets1/

Keywords alongside with disjoint sets for training, validation, and testing given by line IDs used, for example, in:

- A. Fischer, A. Keller, V. Frinken, and H. Bunke: "Lexicon-Free Handwritten Word Spotting Using Character HMMs," in Pattern Recognition Letters, Volume 33(7), pages 934-942, 2012.

- A. Fischer, E. Indermühle, V. Frinken, and H. Bunke: "HMM-Based Alignment of Inaccurate Transcriptions for Historical Documents," in Proc. 11th Int. Conf. on Document Analysis and Recognition, pages 53-57, 2011.

* sets2/

Disjoint sets for training, validation, and testing given by word IDs used, for example, in:

- A. Fischer, K. Riesen, and H. Bunke: "Graph Similarity Features for HMM-Based Handwriting Recognition in Historical Documents," in Proc. 12th Int. Conf. on Frontiers in Handwriting Recognition, pages 253-258, 2010.

- A. Fischer, M. Wüthrich, M. Liwicki, V. Frinken, H. Bunke, G. Viehhauser, and M. Stolz: "Automatic Transcription of Handwritten Medieval Documents," in Proc. 15th Int. Conf. on Virtual Systems and Multimedia, pages 137–142, 2009.

* data/page_images

300dpi JPG images of the original manuscript pages.

* data/line_images_normalized

PNG images of binarized and normalized (skew and height) text lines.

* data/word_images_normalized

PNG images of binarized and normalized (skew and height) words.

* ground_truth/transcription.txt

[lineID] [word_1]|...|[word_n]

Text line transcription with word spellings given by:

[character_1]-...-[character_m]

Besides lower case and upper case letters, two punctuation marks, namely period ("pt") and dash ("eq"), are used as well as several specially encoded characters that are specific to the medieval German language and the Gothic script ("ha115", "hc097", etc).

* ground_truth/word_labels.txt

Word image labels. Note that the periods ("pt") are not provided as word images.

======= CONTACT INFORMATION ======= 

If you have any questions or suggestions, please contact Andreas Fischer (afischer@iam.unibe.ch).
