python main.py main \
--data_name='mnist' \
--data_dir='./data/' \
--img_height=28 \
--img_width=28 \
--img_channels=1 \
--noise_dimension=100 \
--noise_distribution='gaussian' \
--model_name='kdforgan' \
--alpha_kdforgan=0.3 \
--scale=0.2 \
--epochs=2 \
--batch_size=64 \
--plot_interval=10 \
--save_model_dir='./checkpoints/' \
--only_test=False \
--save_res_dir='./results/' \
--num_vis=25