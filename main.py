import os.path
import os
import streamlit as st
import data_prep


def main():
    st.set_page_config(page_title='Deepfake Detection', layout='wide')
    VIDEO_EXTENSIONS = ['mp4', 'ogv', 'm4v', 'webm', 'avi']
    faces_path = './imagery/faces/'
    frames_path = './imagery/frames/'
    boxed_frames_path = './imagery/boxed_frames/'
    video_path = './imagery/video/'
    zipped_path = './imagery/zipped/'
    arrow_image = './imagery/arrow-64.png'
    zip_file = 'prediction.zip'
    uploaded_video_encoded_H264 = 'uploaded_video_encoded_H264.mp4'
    prediction_video_encoded_H264 = 'prediction_video_encoded_h264.mp4'

    header = st.container()
    start = st.container()
    predict = st.container()
    result = st.container()

    with header:
        st.title('Deepfake detection')
        st.header('Explanation')
        st.markdown('Deepfakes have been an emerging problem in recent years. It has become difficult to tell the difference between what is real and what is fake. This technology is already being abused in various industries, politics, celebrities, war.... How can you trust videos on social media if the person in the video can be manipulated? This is where this app comes in. You can check the videos you find online for Deepfakes. ')

    with start:
        st.header('Get started')

        # section where user can upload a video
        uploaded_video = st.file_uploader('Select a video the you want to check for deepfake.',
                                          type=VIDEO_EXTENSIONS)  # upload the video]
        
        data_prep.delete_files_in_dir(faces_path)  # delete all the saved files from previous video
        data_prep.delete_files_in_dir(frames_path)
        data_prep.delete_files_in_dir(boxed_frames_path)
        data_prep.delete_files_in_dir(video_path)
        data_prep.delete_files_in_dir(zipped_path)

        if uploaded_video is not None:

            # save the uploaded video
            with open(os.path.join(video_path, uploaded_video.name), 'wb') as f:  
                f.write(uploaded_video.getbuffer())

            # convert the uploaded video file to h264 codec
            os.system('ffmpeg -i ' + os.path.join(video_path, uploaded_video.name) + ' -vcodec libx264 -acodec aac ' + video_path + uploaded_video_encoded_H264)
            
            # displaying the converted uploaded video file
            video_file = open(video_path + uploaded_video_encoded_H264, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            st.success('Video successfully uploaded.')
            

            with predict:
                st.header('Processing')

                progress_bar = st.progress(0)
                latest_iteration = st.empty()
                progress_bar.progress(1)

                col1, col2, col3, col4, col5  = st.columns(5)

                latest_iteration.text('step 1/6     Generating images from video frames.    Please be patient, this may take a while.')
                fps = data_prep.generate_images_from_videos(video_path + uploaded_video_encoded_H264, frames_path)
                progress_bar.progress(20)

                with col1:
                    st.image(data_prep.get_first_image_from_dir(frames_path))

                latest_iteration.text('step 2/6     Cropping the found faces.   Please be patient, this may take a while.')
                data_prep.pull_faces_from_images(frames_path, faces_path)
                progress_bar.progress(40)

                with col2:
                    st.image(arrow_image)

                with col3:
                    st.markdown('')
                    st.image(data_prep.get_first_image_from_dir(faces_path))

                latest_iteration.text('step 3/6     Making predictions on found faces   Please be patient, this may take a while.')
                prediction = data_prep.predict_on_faces(faces_path)
                progress_bar.progress(60)

                latest_iteration.text('step 4/6     Defining bounding boxes colors  Please be patient, this may take a while.')
                box_color_list = data_prep.define_box_color(prediction)
                progress_bar.progress(80)

                latest_iteration.text('step 5/6     Drawing bounding boxes on images    Please be patient, this may take a while.')
                data_prep.draw_bounding_boxes_on_images(frames_path, boxed_frames_path, box_color_list)
                progress_bar.progress(90)

                with col4:
                    st.image(arrow_image)

                with col5:
                    st.image(data_prep.get_first_image_from_dir(boxed_frames_path))

                latest_iteration.text('step 6/6     Compiling boxed frames into a new video.    Please be patient, this may take a while.')
                data_prep.compile_bounding_box_video(boxed_frames_path, video_path, 30)
                progress_bar.progress(100)

                latest_iteration.text('')
                st.success('All done.')

            with result:

                st.header('Prediction video')

                # convert the prediction video file to h264 codec
                os.system('ffmpeg -i ' + os.path.join(video_path, 'prediction.avi') + ' -vcodec libx264 -acodec aac ' + video_path + prediction_video_encoded_H264)

                # displaying the converted prediction video file
                video_file = open(video_path + prediction_video_encoded_H264, 'rb')
                video_bytes = video_file.read() 
                st.video(video_bytes) 

                # zip the predictin video
                os.system('zip -r ' + os.path.join(zipped_path, zip_file) + ' ' + os.path.join(video_path, 'prediction.avi'))

                # download prediction video
                with open(os.path.join(zipped_path, zip_file), 'rb') as f:
	                st.download_button(
                        label="Download video",
                        data=f,
                        file_name="prediction.zip",
                        mime="application/zip"
                    )


if __name__ == "__main__":
    main()
