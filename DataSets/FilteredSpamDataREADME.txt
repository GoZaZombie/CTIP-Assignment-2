## Overview
This dataset contains emails labeled as either **ham (legitimate)** or **spam (junk email)**.

- **Columns**  
  - `text`: The content of the email  
  - `label`: Label ham, or spam 
  - `URL`: If the message contains a URL
  - `Email`: if the message contains a email
  - `PHONE`: if the message contaisn a phone number
  - `lang`: what language it is in

## Labels
- Ham: Normal, non-spam emails  
- Spam: Junk or malicious emails  


NOTE: 
this document was process using the datacleaning.py file found in this project and isn't the original iteration. When training, the columns may of been renamed or slightly alted when being added to the model. 



Original copy: DirtySMSData.csv

Sourced: https://github.com/vinit9638/SMS-scam-detection-dataset/blob/main/sms_scam_detection_dataset_merged_with_lang.csv  
