# Create a new project with several tasks and brush preannotations
# Contributed by https://github.com/berombau:
# https://github.com/heartexlabs/label-studio-sdk/issues/19#issuecomment-992327281


from sys import argv

LABEL_STUDIO_API_KEY = '34d8eedc4097b203ad63b659530027a29f3987be'
#IMAGE = argv[0]
RLE_FILE = argv[1]



import numpy as np
import label_studio_converter.brush as brush
from label_studio_sdk import Client
import skimage

LABEL_STUDIO_URL = 'http://localhost:8080'
LABEL = 'Cell'


ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
ls.check_connection()

project = ls.start_project(
    title=LABEL,
    label_config=f"""
    <View>
    <Image name="image" value="$image" zoom="true"/>
    <BrushLabels name="brush_labels_tag" toName="image">
        <Label value="{LABEL}" background="#ff0000"/>
    </BrushLabels>
    </View>
    """,
)

task_ids = project.import_tasks(
    [{'image': f'/data/upload/2/4425adf9-PC-10x-wide.jpg'}]
)


rles = eval(open(RLE_FILE, 'r').read())

for rle in rles:

  project.create_prediction(
      task_id=task_ids[0],
      model_version='µSAM/vit_h',
      result=[
          {
              "from_name": "brush_labels_tag",
              "to_name": "image",
              "type": "brushlabels",
              'value': {"format": "rle", "rle": rle, "brushlabels": [LABEL]},
          }
      ],
  ) 