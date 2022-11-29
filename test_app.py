from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("It is a bad and good day!"))

# exception: No module named 'keras.saving.hdf5_format'
# because of an older keras version
# /home/mirevi/janina/gpt_j_test_app/venv/lib/python3.10/site-packages/transformers/modeling_tf_utils.py 
# comment out: #from keras.saving.hdf5_format import save_attributes_to_hdf5_group

classifier = pipeline("zero-shot-classification")
print(classifier("It is a bad and good day!",
        candidate_labels="