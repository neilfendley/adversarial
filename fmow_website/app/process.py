#This is the file to compile and create the model
import sys
import redis
import os
import json
sys.path.append('./data_ml_functions/DenseNet')
from keras.models import Model, load_model, Sequential
import numpy as np

from data_ml_functions.mlFunctions import get_cnn_model, img_metadata_generator,get_lstm_model,codes_metadata_generator
 

def main():
#    os.environ['REDIS_HOST'] = ''
#    os.environ['REDIS_DB'] = ''
#    os.environ['REDIS_CHANNEL'] = ''
#    os.environ['REDIS_OUTPUT_CHANNEL'] = ''
#    os.environ['REDIS_PORT'] = '2002'
    connection = redis.StrictRedis(host=os.environ['REDIS_HOST'], port=os.environ['REDIS_PORT'], db=os.environ['REDIS_DB'])
    pubsub = connection.pubsub()
    pubsub.psubscribe([os.environ['REDIS_CHANNEL']])
    frame_buffer = []
    model = Sequential()
    model = load_model('cnn_image_only.model')
    
    for message in pubsub.listen():  # Listen for messages
        if 'message' in message['type']: # Some other PUBSUB messages will flow by that you _should_ignore_.  This matches 'pmessage' or 'message'

            decoded_message = json.loads(message['data'])  # Decode the JSON message to a Python dict

            stream_id = decoded_message['stream_id']
            frame_id = decoded_message['frame_id']

            frame_bytes = connection.get(frame_id)  # Grab the frame bytes
            frame_buffer.append(frame_bytes)
            frame_metadata = connection.get(frame_id + "metadata")  # Grab the frame metadata
            data = np.expand_dims(frame_bytes, axis=0)
            pred = model.predict(data, batch_size=data.shape[0])
            return_json = {}
            metadata = {} 
            metadata['preds'] = pred
            return_json['stream_id'] = stream_id
            return_json['frame_id'] = frame_id
            return_json['processor'] = "vortex.processors.edu.jhuapl.fmow_website"
            return_json['metadata'] = [metadata]
            connection.publish(os.environ['REDIS_OUTPUT_CHANNEL'], json.dumps(return_json))
    
    


if __name__ == '__main__':
     main()
