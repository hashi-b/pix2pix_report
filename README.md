# pix2pix_report
eizou_media report

結果が概ね正しいか確認できるよう実装はchainerのcodeが存在するfacadeを選択しました

・実行方法

facade_datasetをhttp://cmp.felk.cvut.cz/~tylecr1/facade/から落としbase直下にjpg及びpngを置く

run_Unet.sh $1 ($1は結果の出力先フォルダ名) 　　　でUnet(論文中におけるL1のみ)による変換モデルの学習及び変換　GPU環境で2時間程度

run_pix2pix.sh $1 ($1は結果の出力先フォルダ名) でcGAN(論文中におけるcGAN+L1)による変換モデルの学習及び変換 GPU環境で半日程度

sh内でepochやbatch、gpu使用の有無などのオプションも変更可能


画像データのIO周りに関しては以下のコードを参考にしました

Copyright (c) 2016 Eiichi Matsumoto

