import face_recognition
import cv2


input_movie = cv2.VideoCapture("clip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

# Load some sample pictures and learn how to recognize them.

dp_image = face_recognition.load_image_file("DominicPurcell.png")
dp_face_encoding = face_recognition.face_encodings(dp_image)[0]

wm_image = face_recognition.load_image_file("WentworthMiller.png")
wm_face_encoding = face_recognition.face_encodings(wm_image)[0]

wf_image = face_recognition.load_image_file("WilliamFichtner.png")
wf_face_encoding = face_recognition.face_encodings(wf_image)[0]

lg_image = face_recognition.load_image_file("LaneGarrison.png")
lg_face_encoding = face_recognition.face_encodings(lg_image)[0]

ps_image = face_recognition.load_image_file("PeterStormare.png")
ps_face_encoding = face_recognition.face_encodings(ps_image)[0]

ww_image = face_recognition.load_image_file("WadeWilliams.png")
ww_face_encoding = face_recognition.face_encodings(ww_image)[0]

sk_image = face_recognition.load_image_file("StacyKeach.png")
sk_face_encoding = face_recognition.face_encodings(sk_image)[0]

rt_image = face_recognition.load_image_file("RobinTunney.png")
rt_face_encoding = face_recognition.face_encodings(rt_image)[0]

swc_image = face_recognition.load_image_file("SarahWayneCollies.png")
swc_face_encoding = face_recognition.face_encodings(swc_image)[0]

rk_image = face_recognition.load_image_file("RobertKnepper.png")
rk_face_encoding = face_recognition.face_encodings(rk_image)[0]

an_image = face_recognition.load_image_file("AmauryNolasco.png")
an_face_encoding = face_recognition.face_encodings(an_image)[0]

known_faces = [
    wm_face_encoding,
    dp_face_encoding,
    wf_face_encoding,
    lg_face_encoding,
    ps_face_encoding,
    ww_face_encoding,
    sk_face_encoding,
    rt_face_encoding,
    swc_face_encoding,
    rk_face_encoding,
    an_face_encoding,


]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Wentworth Miller"
        elif match[1]:
            name = "Dominic Purcell"
        elif match[2]:
            name = "William Fichtner"
        elif match[3]:
            name = "Lane Garrison"
        elif match[4]:
            name = "Peter Stormare"
        elif match[5]:
            name = "Wade Williams"
        elif match[6]:
            name = "Stacy Keache"
        elif match[7]:
            name = "Robin Tunney"
        elif match[8]:
            name = "Sarah Wayne Callies"
        elif match[9]:
            name = "Robert Knepper"
        elif match[10]:
            name = "Amaury Nolasco"
        else:
            name = "Unknown"


        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
