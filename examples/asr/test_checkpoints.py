import os
import jiwer
import gspread
import sys, getopt

from nemo.collections.asr.models import EncDecCTCModel

def eval(doc1, doc2):
  transform = jiwer.Compose([
      jiwer.RemovePunctuation(),
      jiwer.RemoveMultipleSpaces(),
      jiwer.ToLowerCase()
  ])

  hypothesis = transform(doc1)
  ground_truth = transform(doc2)

  measures = jiwer.compute_measures(ground_truth, hypothesis)

#   print('Evaluation:')
  scores = {
      'Substitutions': measures['substitutions'],
      'Deletions': measures['deletions'],
      'Insertions': measures['insertions'],
      'Total Error Rate': measures['wer'],
  }
#   print(data)
#   print('----------------------------------------')

  return scores['Total Error Rate']

def test(checkpoint_path):
    scores = []
    asr_model = EncDecCTCModel.load_from_checkpoint(checkpoint_path)
    test_files = [file.split('.')[0].split('/')[-1] for file in os.listdir('/home/visha/NeMo_test/examples/asr/test_files') if file.split('.')[-1] == 'wav']
    for file in test_files:
        asr_out = asr_model.transcribe([f"/home/visha/NeMo_test/examples/asr/test_files/{file}.wav"])[0]
        truth = open(f"/home/visha/NeMo_test/examples/asr/test_files/{file}.txt").read()

        scores.append(eval(asr_out, truth))

    return scores

def update_sheet(scores, checkpoint_path, name):
    gc = gspread.service_account(filename='/home/visha/NeMo_test/examples/asr/desicrew-v1-088082cf46f3.json')
    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1-OUI0RkDLE0OfOVS0a1PqBTNfGP0fjTMHkJWMUnoj9U/edit#gid=0")
    sheet = sh.worksheet('Sheet2')

    data = sheet.get_all_values()[1:]
    index = len(data)


    sheet.update_cell(index+2, 1, str(name))
    sheet.update_cell(index+2, 2, str(checkpoint_path))
    for i, score in enumerate(scores):
        sheet.update_cell(index+2, i+2, score)
    print('Sheet updated!')

def main(argv):
    log_directory = sys.argv[1]

    checkpoints = [checkpoint for checkpoint in os.listdir(str(log_directory) + '/checkpoints') if checkpoint.split('.')[-1] == "ckpt"]
    if len(checkpoints) == 0:
        checkpoints = [checkpoint for checkpoint in os.listdir(str(log_directory) + '/checkpoints') if checkpoint.split('.')[-1] == 'nemo']

    for checkpoint_path in checkpoints:
        update_sheet(test(str(log_directory) + '/checkpoints/' + checkpoint_path), checkpoint_path, log_directory)

if __name__ == '__main__':
    main(sys.argv)