import pandas as pd


SMSSpamData2 = pd.read_csv(r'DataSets/DirtySMSData.csv',  encoding='latin-1')


SMSSpamData2 = SMSSpamData2[SMSSpamData2['lang'] == 'en']

SMSSpamData2.to_csv('FilteredSpamData.csv', index=False)