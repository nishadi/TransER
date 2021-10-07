from transer import model

src_dataset = 'dblp-acm1'
src_link = 'A-A'
tgt_dataset = 'dblp-scholar1'
tgt_link = 'A-A'


model.predict(src_dataset, src_link, tgt_dataset, tgt_link)
