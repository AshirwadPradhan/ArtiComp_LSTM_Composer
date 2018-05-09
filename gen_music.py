from mido import MidiFile, MidiTrack, Message
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
import numpy
import time

class ArtiComp():

    def __init__(self):
        self._timesteps = 30
        self._seed = []
        self._max_time = 1

    def _load_dataset(self):
        '''
        loads the midi file
        converts the midi file into notes based only on 'notes_on'
        '''
        # load the midi file
        midi_song = MidiFile('data/beethoven_opus10_1.mid')
        notes = []

        # for msg in midi_song:
        #     print(str(msg).encode('utf-8'))

        now_time = float(0)
        prev_time = float(0)


        # get the dataset
        for msg in midi_song:
            # getting the current time and converting ticks to sec
            now_time += msg.time
            
            # check to see if the msg is not metadata
            if not msg.is_meta:
                # only considering one channel here
                if msg.channel == 0:
                    # note on represents actual notes data
                    if msg.type =='note_on':
                        # save the notes vector
                        # notes vector consists of type, notes, velocity
                        note = msg.bytes()

                        # trim notes to get only notes and velocity data
                        # remove the type data
                        note = note[1:3]

                        # set the duration of note in the note vector
                        note.append(now_time - prev_time)
                        # update the time to keep track of how much time has 
                        # passed
                        prev_time = now_time
                        # add note the notes sequence
                        notes.append(note)
        return notes


    def make_dataset(self):
        '''
        This function has three operations
        1-> loads the midi file
        2-> retrieve notes from the midi file in a particular format [notes, velocity, time] vector
        3-> take the notes vector and scale it
        4-> prepare the time series data based on the timesteps
        
        Output: X (train X), Y(train Y)
        '''
        notes = self._load_dataset()
        # preprocess the data
        t = []
        for note in notes:
            # scaling the notes and velocity on the 128 scale(0 - 127)
            note[0] = note[0]/127
            note[1] = note[1]/127
            t.append(note[2])
        # getting the max time interval
        self._max_time = max(t)
        for note in notes:
            # scaling the time based on max time
            note[2] = note[2]/self._max_time

        # getting data in the right format
        X = []
        Y = []
        
        for i in range(len(notes)-self._timesteps):
            # x has timesteps from first 0 to 29
            x = notes[i:i+self._timesteps]
            # y has the 30th prediction note 
            y = notes[i+ self._timesteps ]

            # Feature
            X.append(x)
            # Label
            Y.append(y)

        self._seed = notes[0:self._timesteps]

        X = numpy.array(X)
        Y = numpy.array(Y)
        # print(X.ndim)
        return (X, Y)

    def build_model(self, X, Y):
        '''
        builds a two layer LSTM model based on the training vectors provided
        Input : X should be in numpy.ndarray(30, 3)
        Input : Y should be in numpy.ndim(1)

        Returns a Keras model
        '''
        # building a LSTM model
        print('Building model...\n')
        model = Sequential()
        model.add(LSTM(128, input_shape=(self._timesteps, 3), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, input_shape=(self._timesteps, 3), return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(3))
        model.add(Activation('linear'))

        # Compiling Model
        print('Compiling Model...\n')
        model.compile(loss='mse', optimizer='rmsprop')

        # Fitting Model
        print('Fitting Model...\n')
        model.fit(X, Y, batch_size=250, epochs=200, verbose=1)

        return model


    def generate_music(self, model, length=3000):
        '''
        Generates the midi file based on the learning
        model - trained model
        length - length of the midi sequence
        '''
        # Generating the music
        # Making predictions
        tic = time.time()
        y_pred = []
        x = self._seed
        x = numpy.expand_dims(x, axis=0)

        print('Making Music...')
        for _ in range(length):
            pred = model.predict(x)
            x = numpy.squeeze(x)
            x = numpy.concatenate((x, pred))
            x = x[1:]
            x = numpy.expand_dims(x, axis=0)
            pred = numpy.squeeze(pred)
            y_pred.append(pred)

        print('Compiling Music File...')
        for p in y_pred:
            
            # Rescaling the value to 0 - 127
            # and ensuring it's a valid midi file
            p[0] = int(127*p[0])
            if p[0] < 0:
                p[0] = 0
            elif p[0] > 127:
                p[0] = 127

            p[1] = int(127*p[1])
            if p[1] < 0:
                p[1] = 0
            elif p[1] > 127:
                p[1] = 127
            # Rescaling the time back to normal time
            p[2] *= self._max_time
            if p[2] < 0:
                p[2] = 0
        # print(y_pred)

        # rendering midi file
        print('Rendering Midi File...')
        pred_mid_song = MidiFile()
        track = MidiTrack()
        pred_mid_song.tracks.append(track)

        for p in y_pred:
            # appending other info as channel(0) and type(147)
            p = numpy.insert(p, 0, 147)

            byte = p.astype(int)
            msg = Message.from_bytes(byte[0:3])
            _time = int(p[3]/0.001025)
            msg.time = _time
            track.append(msg)

        print('Saving midi file')
        pred_mid_song.save('out/beth_gen1.midi')
        toc = time.time()
        print('Time taken for rendering midi file {}'.format(toc-tic))
        print('Done')


if __name__ == '__main__':
    o_articomp = ArtiComp()
    X, Y = o_articomp.make_dataset()
    model = o_articomp.build_model(X, Y)
    o_articomp.generate_music(model)