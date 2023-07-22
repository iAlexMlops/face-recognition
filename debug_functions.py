import face_recognition
import pandas as pd


# Load a sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("images/biden.jpg")
biden_face_locations = face_recognition.face_locations(biden_image)
print(f"Biden: {biden_face_locations}")

forest_image = face_recognition.load_image_file("images/forest.jpg")
forest_face_locations = face_recognition.face_locations(forest_image)
print(f"Forest: {forest_face_locations}")

alex_image = face_recognition.load_image_file("images/alex.jpg")
alex_face_locations = face_recognition.face_locations(alex_image)
print(f"Alex: {alex_face_locations}")


# Sample of face encodings fanctionality
biden_encoding = face_recognition.face_encodings(biden_image)
print(f"Biden Encoding Len: {len(biden_encoding)}")
print(f"Biden Encoding: {biden_encoding[0].shape}")

forest_encoding = face_recognition.face_encodings(forest_image)
print(f"Forest Encoding Len: {len(forest_encoding)}")
print(f"Forest Encoding: {forest_encoding}")

alex_encoding = face_recognition.face_encodings(alex_image)
print(f"Alex Encoding Len: {len(alex_encoding)}")
print(f"Alex Encoding: {alex_encoding[0].shape}")


# Compare the encodings
result = face_recognition.compare_faces(known_face_encodings=[biden_encoding[0],
                                                              alex_encoding[0]
                                                              ],
                                        face_encoding_to_check=alex_encoding[0]
                                        )
print(f"Result: {result} \n\n")


# Test group photo encodings
group_image = face_recognition.load_image_file("images/group.jpg")
group_face_locations = face_recognition.face_locations(group_image)
print(f"Group: {group_face_locations}")

group_encoding = face_recognition.face_encodings(group_image)
print(f"Group Encoding Len: {len(group_encoding)}")
print(f"Group Encoding 0: {group_encoding[0].shape}")
print(f"Group Encoding Sample: {group_encoding[0]}")

df = pd.DataFrame(group_encoding, index=["Children", "Adults", "Biden"])
print(df.head())

df.to_parquet("datasets/group_encoding.parquet")


# Output
#
# Biden: [(241, 740, 562, 419)]
# Forest: []
# Alex: [(201, 379, 468, 111)]
# Biden Encoding Len: 1
# Biden Encoding: (128,)
# Forest Encoding Len: 0
# Forest Encoding: []
# Alex Encoding Len: 1
# Alex Encoding: (128,)
# Result: [False, True]
#
#
# Group: [(139, 262, 325, 77), (118, 902, 304, 716), (43, 563, 266, 340)]
# Group Encoding Len: 3
# Group Encoding 0: (128,)
# Group Encoding Sample: [-0.13621327  0.10461364  0.09946977 -0.05099441 -0.1718619  -0.00225095
#  -0.03488863 -0.05527756  0.17689347 -0.01048833  0.23332793 -0.05320296
#  -0.24220094 -0.04723439 -0.0220543   0.19289561 -0.11227828 -0.10088836
#  -0.08047725 -0.00849514 -0.02233211  0.06735176 -0.0091906   0.04126503
#  -0.14952359 -0.3774032   0.00522227 -0.20484127 -0.09034564 -0.06748568
#  -0.03105492  0.08088478 -0.13284832 -0.05534847  0.01927603  0.09094137
#  -0.13445139 -0.04714725  0.21764557  0.05587866 -0.20707679  0.04973353
#   0.02394059  0.28047907  0.17386898  0.00245022  0.06276345 -0.05119509
#   0.20568115 -0.27347329  0.04058605  0.10311396  0.12587163  0.03912607
#   0.1169234  -0.10968411  0.0564149   0.12036245 -0.31395233  0.11401353
#   0.05495398 -0.08385953 -0.02043144 -0.03995985  0.23915051  0.14133215
#  -0.07321248 -0.051184    0.17951408 -0.11904829 -0.07668346  0.06895245
#  -0.12846963 -0.20448217 -0.30235785  0.00299951  0.42017627  0.08609452
#  -0.20227042 -0.04574115 -0.04070884 -0.08287279  0.04010998  0.04045628
#  -0.04527701 -0.08400288 -0.09290014 -0.02259386  0.2896983  -0.03999954
#   0.01824536  0.27610165  0.05322869 -0.01826592  0.00302738 -0.00756494
#  -0.06780951 -0.04107308 -0.08499713 -0.0228783  -0.01459473 -0.16229501
#  -0.01368727  0.03431561 -0.28135863  0.13078064  0.01704703  0.00112088
#  -0.04152203 -0.00594999 -0.10903687  0.01159319  0.21970886 -0.21470329
#   0.23958178  0.13414885  0.07469991  0.10697526  0.00866418 -0.05506641
#  -0.04697538 -0.05202758 -0.15248007 -0.10039301  0.07353676 -0.02018823
#   0.05277465  0.11039373]
