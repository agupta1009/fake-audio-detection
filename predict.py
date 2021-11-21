from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np
import tensorflow
from PIL import Image, ImageChops
import os 
import cv2
import glob
import soundfile as sf
import scipy.io.wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import shutil



from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img




def ELA(img_path):

    """Performs Error Level Analysis over a directory of images"""
    
    TEMP = 'ela_' + 'temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)
    except:
        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)
        
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    return diff

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detectFaces(path,f,dest):
  
    filenames = glob.glob(path+'/*.mp4')
    k=0

    for filename in filenames:
  
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
    nof = v_len//f
  
    j=0
    cnt = 0
    while(cnt<f and j<v_len):

        v_cap.set(1,j)
        # Load frame
        success, img = v_cap.read()
        if not success:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = faces[:1]
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display
    
        cv2.imwrite(dest +'/'+ str(k)+'.jpg', img[y:y+h,x:x+w])
        k+=1
        j+=nof
        cnt+=1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

  
    v_cap.release()
    cv2.destroyAllWindows()

def prepare_image(file):

    img=image.load_img(file,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array_expanded_dims=np.expand_dims(img_array,axis=0)
    return tensorflow.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

app = Flask(__name__)
uploaded_file = ''

app.config['image_upload'] = 'D:/ankush/projects/itsfake/uploads'
app.config['video_upload'] = 'D:/ankush/projects/itsfake/video_upload'
app.config['face'] = 'D:/ankush/projects/itsfake/face'
app.config['audio_upload'] = 'D:/ankush/projects/itsfake/audio_upload'

@app.route('/main',methods = ['GET','POST'])
def main():
    return render_template('main.html')


@app.route('/upload_file' , methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['data']
        data = request.form['fav_language']
        if (data=="image_"):
            uploaded_file.save(os.path.join(app.config['image_upload'],uploaded_file.filename))
            model = load_model('new_model_casia.h5')
            numpydata=np.array(ELA(uploaded_file).resize((128, 128))).flatten() / 255.0
            numpydata = np.resize(numpydata,(1,128,128,3))
            pred = model.predict(numpydata)     
            print("prediction for image:",pred)
            return render_template('./after.html',data = pred[0][0])
        elif (data== "deepfake_"):

            model2 = load_model('deep_fakes.h5')
            uploaded_file.save(os.path.join(app.config['video_upload'],uploaded_file.filename))
            detectFaces(app.config['video_upload'],20,app.config['face'])
            
            count=0
            files = os.listdir(app.config['face'])
            for file in files:
                preprocessed_image=prepare_image(app.config['face']+'/'+file)
                pred = model2.predict(preprocessed_image)
                print(pred)
                if pred[0][0] >= 0.5:
                    count+=1
                if count>10:
                    return render_template('./after.html',data = 0.1)    
                else:
                    return render_template('./after.html',data = 0.9)
        elif (data== "audio_"):

            model3 = load_model('fake_audio.h5')
            uploaded_file.save(os.path.join(app.config['audio_upload'],uploaded_file.filename))
            for filename in os.listdir(f'audio_upload'):
                y, sr = sf.read(f'audio_upload/{filename}')
                plt.specgram(y,Fs=sr);
                plt.axis('off');
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                dir = 'audio_upload_spectogram'
                for f in os.listdir(dir):
                    os.remove(os.path.join(dir, f))
                plt.savefig(f'audio_upload_spectogram/{filename[0:-5]}.png')
                plt.clf()

            filename1 = os.listdir("audio_upload_spectogram/")

            category1 = []
            # Get filename and assign label '0' for real images and stored in dataframe
            for filename in filename1:
                category1.append(1)    
            df3 = pd.DataFrame({'filename': filename1,'category': category1 })  
            df3["category"] = df3["category"].replace({0: 'Real', 1: 'Spoof'})

            test_datagen1 = ImageDataGenerator(rescale=1./255)

            IMAGE_WIDTH=128
            IMAGE_HEIGHT=128
            IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
            batch_size=15
            
            test_generator1 = test_datagen1.flow_from_dataframe(
                df3, 
                "Spectograms",
                x_col='filename',
                y_col='category',
                seed=30,
                target_size=IMAGE_SIZE,
                class_mode='categorical',
                batch_size=batch_size
            )

            pred = model3.predict(test_generator1)

            print(pred)
            if pred[0][0] >= 0.70:
                return render_template('./after.html',data = 0.9)
            else:
                return render_template('./after.html',data = 0.1)
            
        else:
                return render_template('./after.html',data=0.0)

if __name__ == '__main__':
    app.run(debug=True)