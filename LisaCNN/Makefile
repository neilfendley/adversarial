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


# Check out the cleverhans project, if it was not already
$(CLEVERHANS_DIR) :
	git clone https://github.com/tensorflow/cleverhans.git


# This target either:
#   1.  Trains the LISA-CNN, if the weights do not exist locally or
#   2.  Runs some adversarial attacks on a previously trained network.
# 
lisa-cnn : $(CLEVERHANS_DIR)
	PYTHONPATH=$(CLEVERHANS_DIR) python ./lisacnn.py --data_dir=$(LISA_DIR) 

# remove AE (e.g. so they can be re-computed)
clean :
	\rm -rf ./output/Images ./output/Deltas ./output/*png

# kills all output, including the CNN weights
bare : 
	\rm -rf ./output ./lisa_data.npz

# creates a tar file with all images
tar :
	cd output && tar cvf ../images.tar ./images
