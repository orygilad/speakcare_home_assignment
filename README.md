SPEAKCARE Home Assignment
-------------------------

Model choice
------------
the speechbrain_ecapa model was chosen. the model was optimized for text-independent speaker embedding extraction, trained on thousands of different speakers.
the model was trained on input segments of length 2-3 seconds at 16Khz sampling rate.

decoding .m4a files
-------------------
the project decode the provided .m4a and converts them to .wav at the begining. the conversion is done by the pydub library.
the initial sampling rate of the .m4a files is 44.1Khz. during the conversion a resampling to 16Khz is performed.

Algorithmic pipeline
--------------------
The process is as follows:
- Divide to segments and embed:
	Each speaker’s reference sample is divided into a 3 seconds segments. The same for the session test sample.
	embedding s are calculated for each segments based on a provided model.
- Segments similarity calculation & thresholding:
	In each target’s segment, a cosine similarity is calculated with all segments of all speakers.
	In each speaker’s segments a MAX operator takes only a single cosine similarity value giving a length 6 (No. of optional speakers) vector of similarities per target’s segment.
	Segment with a maximum (out of 6) similarity value smaller than a configurable threshold is being filtered out assuming there is no speech in those segments.
- Speaker prediction:
	On the list of filtered segments similarities scores - a probability values are derived by applying a softmax on the similarities along the speaker axis.
	All the probabilities are being averaged along the target’s filtered segments axis. A variance for each speaker’s probability is computed as well.
	The speaker with highest probability is being declared.
- Confidence estimation:
	Confidence is being computed based on a bayesian hypothesis testing, with approximation of taking only the top 2 predicted speakers. the two distributions are assumed to be independent and gaussian with a mean and variance taken upon all session segments.

results
-------
in the provided session, the speaker is Nurse 2 (by ear)
my algorithm predicted correctly that the speaker is Nurse 2 with Confidence: 0.8915 (number between 0-1)
the probability vectors are:
  Nurse1          → 0.154 +- 0.01183381
  Nurse2          → 0.204 +- 0.01083766
  Nurse3          → 0.153 +- 0.00960425
  Nurse4          → 0.164 +- 0.00875090
  Nurse5          → 0.172 +- 0.01468099
  Nurse6          → 0.153 +- 0.00823377


How to run
----------
1. run the following command from project root: pip install -r requirements.txt
2. place the reference speakers files under PROJECT_PATH/data/MY_SPEAKER_FOLDER. place session file under PROJECT_PATH/data/test_audio.
3. run the following command from project root:  python -m utils.convert_all_m4a
4. run the following command: python main.py --speaker_reference_folder my_speakers_folder/ --session my_session --out my_results
5. all required results will be both logged and presented in the console.
6. for your convenience, the scripts compute_embeddings.py can run to offline compute and save audio embeddings. run the following command from project root right after step 2: python -m utils.compute_embeddings
7. after step 6 , main_embeddings.py can be used instead of main.py seemlessly, and produce results much faster.

