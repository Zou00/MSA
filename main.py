import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./utils')
sys.path.append('./utils/APIs')

import torch
import datetime

import argparse
from Config import config
from utils.common import data_format, read_from_file, train_val_split, save_model, write_to_file
from utils.DataProcess import Processor
from Trainer import Trainer

# 确保输出目录存在
if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)
    print(f"创建输出目录: {config.output_path}")

# args
parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true', help='训练模型')
parser.add_argument('--text_pretrained_model', default='roberta-base', help='文本分析模型', type=str)
parser.add_argument('--fuse_model_type', default='OTE', help='融合模型类别', type=str)
parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=1e-2, help='设置权重衰减', type=float)
parser.add_argument('--epoch', default=10, help='设置训练轮数', type=int)

parser.add_argument('--do_test', action='store_true', help='预测测试集数据')
parser.add_argument('--load_model_path', default=None, help='已经训练好的模型路径', type=str)
parser.add_argument('--text_only', action='store_true', help='仅用文本预测')
parser.add_argument('--img_only', action='store_true', help='仅用图像预测')
args = parser.parse_args()
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.bert_name = args.text_pretrained_model
config.fuse_model_type = args.fuse_model_type
config.load_model_path = args.load_model_path
config.only = 'img' if args.img_only else None
config.only = 'text' if args.text_only else None
if args.img_only and args.text_only: config.only = None
print('TextModel: {}, ImageModel: {}, FuseModel: {}'.format(config.bert_name, 'ResNet50', config.fuse_model_type))


# Initilaztion
processor = Processor(config)
if config.fuse_model_type == 'CMAC' or config.fuse_model_type == 'CrossModalityAttentionCombine':
    from Models.CMACModel import Model
elif config.fuse_model_type == 'HSTEC' or config.fuse_model_type =='HiddenStateTransformerEncoder':
    from Models.HSTECModel import Model
elif config.fuse_model_type == 'OTE' or config.fuse_model_type == 'OutputTransformerEncoder':
    from Models.OTEModel import Model
elif config.fuse_model_type == 'NaiveCat':
    from Models.NaiveCatModel import Model
else:
    from Models.NaiveCombineModel import Model
model = Model(config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
trainer = Trainer(config, processor, model, device)


# Train
def train():
    print('开始数据预处理...')
    # 创建日志文件
    log_path = os.path.join(config.output_path, f'{config.fuse_model_type}_training_log.txt')
    
    # 记录训练开始信息
    with open(log_path, 'a', encoding='utf-8') as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"训练开始时间: {timestamp}\n")
        log_file.write(f"模型类型: {config.fuse_model_type}\n")
        log_file.write(f"文本模型: {config.bert_name}, 图像模型: ResNet50\n")
        log_file.write(f"学习率: {config.learning_rate}, 权重衰减: {config.weight_decay}, Epoch数: {config.epoch}\n")
        log_file.write("="*50 + "\n\n")
    
    data_format(os.path.join(config.root_path, './data/train.txt'), 
    os.path.join(config.root_path, './data/data'), os.path.join(config.root_path, './data/train.json'))
    data = read_from_file(config.train_data_path, config.data_dir, config.only)
    train_data, val_data = train_val_split(data)
    train_loader = processor(train_data, config.train_params)
    val_loader = processor(val_data, config.val_params)

    best_avg_f1 = 0
    best_weighted_f1 = 0
    epoch = config.epoch
    for e in range(epoch):
        print('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20)
        tloss, tloss_list = trainer.train(train_loader)
        print('Train Loss: {}'.format(tloss))
        vloss, metrics = trainer.valid(val_loader)
        print('Valid Loss: {}'.format(vloss))
        print('Valid Metrics:')
        print(f'  Accuracy: {metrics["accuracy"]:.4f}')
        print(f'  Average F1: {metrics["avg_f1"]:.4f}')
        print(f'  Weighted F1: {metrics["weighted_f1"]:.4f}')
        
        # 将训练结果写入日志文件
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20 + '\n')
            log_file.write(f'Train Loss: {tloss}\n')
            log_file.write(f'Valid Loss: {vloss}\n')
            log_file.write('Valid Metrics:\n')
            log_file.write(f'  Accuracy: {metrics["accuracy"]:.4f}\n')
            log_file.write(f'  Average F1: {metrics["avg_f1"]:.4f}\n')
            log_file.write(f'  Weighted F1: {metrics["weighted_f1"]:.4f}\n')
        
        # 使用F1值保存最佳模型
        if metrics["avg_f1"] > best_avg_f1:
            best_avg_f1 = metrics["avg_f1"]
            model_name = f"{config.fuse_model_type}_best_avg_f1"
            save_model(config.output_path, model_name, model)
            output_model_dir = os.path.join(config.output_path, model_name)
            output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
            print(f'更新最佳平均F1模型! 保存路径: {output_model_file}')
            # 将模型保存路径写入日志
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f'更新最佳平均F1模型! 保存路径: {output_model_file}\n')
            
        if metrics["weighted_f1"] > best_weighted_f1:
            best_weighted_f1 = metrics["weighted_f1"]
            model_name = f"{config.fuse_model_type}_best_weighted_f1"
            save_model(config.output_path, model_name, model)
            output_model_dir = os.path.join(config.output_path, model_name)
            output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
            print(f'更新最佳加权F1模型! 保存路径: {output_model_file}')
            # 将模型保存路径写入日志
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f'更新最佳加权F1模型! 保存路径: {output_model_file}\n')
                
        print()
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write('\n')

    print(f'训练完成! 最佳平均F1: {best_avg_f1:.4f}, 最佳加权F1: {best_weighted_f1:.4f}')
    # 记录训练结束信息
    with open(log_path, 'a', encoding='utf-8') as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n训练结束时间: {timestamp}\n")
        log_file.write(f"最终结果 - 最佳平均F1: {best_avg_f1:.4f}, 最佳加权F1: {best_weighted_f1:.4f}\n")
        log_file.write("="*50 + "\n\n")


# Test
def test():
    data_format(os.path.join(config.root_path, './data/test_without_label.txt'), 
    os.path.join(config.root_path, './data/data'), os.path.join(config.root_path, './data/test.json'))
    test_data = read_from_file(config.test_data_path, config.data_dir, config.only)
    test_loader = processor(test_data, config.test_params)

    if config.load_model_path is not None:
        model.load_state_dict(torch.load(config.load_model_path))

    outputs = trainer.predict(test_loader)
    formated_outputs = processor.decode(outputs)
    write_to_file(config.output_test_path, formated_outputs)


# main
if __name__ == "__main__":
    if args.do_train:
        train()
    
    if args.do_test:
        if args.load_model_path is None and not args.do_train:
            print('请输入已训练好模型的路径load_model_path或者选择添加do_train arg')
        else:
            test()