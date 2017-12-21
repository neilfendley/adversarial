# Demonstrates how to run the LISA-CNN training codes.


# directory containing cleverhans AE library
CLEVERHANS_DIR=./cleverhans

# directory to subset of LISA dataset containing the
# street sign images (extracted from whole images).
#
# Note that you must manually run the extraction script
# provided with LISA a-priori.
LISA_DIR=/home/pekalmj1/Data/LISA/annotations


default : lisa-cnn


$(CLEVERHANS_DIR) :
	git clone https://github.com/tensorflow/cleverhans.git


# Note that the network need only be trained once.
# Subsequent runs evaluate on the test set.
lisa-cnn : $(CLEVERHANS_DIR)
	PYTHONPATH=$(CLEVERHANS_DIR) python ./train_lisacnn.py --data_dir=$(LISA_DIR) --epsilon=30


# remove AE so they can be re-computed
clean :
	\rm -rf ./output/images

# kills all output, including the CNN weights
bare : 
	\rm -rf ./output
