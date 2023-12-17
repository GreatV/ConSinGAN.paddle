import paddle
from paddle import static
from ConSinGAN import models
from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_name', help=
        'input image name for training', required=True)
    parser.add_argument('--naive_img', help=
        'naive input image  (harmonization or editing)', default='')
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--train_mode', default='generation', choices=[
        'generation', 'retarget', 'harmonization', 'editing', 'animation'],
        help='generation, retarget, harmonization, editing, animation')
    parser.add_argument('--lr_scale', type=float, help=
        'scaling of learning rate for lower stages', default=0.1)
    parser.add_argument('--train_stages', type=int, help=
        'how many stages to use for training', default=6)
    parser.add_argument('--fine_tune', action='store_true', help=
        'whether to fine tune on a given image', default=0)
    parser.add_argument('--model_dir', help=
        'model to be used for fine tuning (harmonization or editing)',
        default='')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    model = models.GrowingGenerator(opt)
    reals_shapes = [[1, 3, 25, 38], [1, 3, 31, 47], [1, 3, 41, 61], [1, 3, 
        60, 89], [1, 3, 115, 171], [1, 3, 167, 250]]
    fixed_noise = [paddle.randn(shape=shape) for shape in reals_shapes]
    noise_amp = [1, 0.1, 0.1, 0.1, 0.1, 0.1]
    try:
        fixed_noise = list(paddle.static.InputSpec.from_tensor(t) for t in fixed_noise)
        reals_shapes = static.InputSpec.from_tensor(paddle.to_tensor(reals_shapes))
        noise_amp = static.InputSpec.from_tensor(paddle.to_tensor(noise_amp)) 
        paddle.jit.save(model, input_spec=(fixed_noise, reals_shapes, noise_amp), path="./model", skip_prune_program=True)
        print('[JIT] paddle.jit.save successed.')
        exit(0)
    except Exception as e:
        print('[JIT] paddle.jit.save failed.')
        raise e