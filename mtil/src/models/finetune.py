import copy
import os
import csv

import clip.clip as clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier, evaluate_2
from .helpers import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation

from collections import defaultdict
import signal, torch, sys

model_ref = None
args_ref = None
iteration_save = 0

gradient_save = False
gradients_per_layer = defaultdict(list)

def should_track_layer(name):
    if name.endswith(".bias"):
        return False
    if "attn" in name and "weight" in name:
        return True

    if "mlp" in name and "weight" in name:
        return True
        
    if "ln" in name and "weight" in name:
        return True

    return False

def save_output_gradients(layer_name):    
    global gradients_per_layer, gradient_save
    
    def hook_fn(grad):
        if gradient_save:
            gradients_per_layer[layer_name].append(grad.detach().cpu().flatten())
    
    return hook_fn


def svd_basis(gradient_per_layer_list, energy=0.97):

    basis_per_layer = {}
    
    for name, gradient_list in gradient_per_layer_list.items():
        
        if len(gradient_list) < 2:
            continue
        q = min(len(gradient_list), 20) 
        G = torch.stack(gradient_list, dim=0)
        G = G.cpu()
        # print(f"{name}: len = {len(gradient_list)}, G.shape = {G.shape}")
        if hasattr(torch.linalg, "svd_lowrank"):
            U, S, V_transpose = torch.linalg.svd_lowrank(G, q=q, niter=4)
        elif hasattr(torch, "svd_lowrank"):
            U, S, V_transpose = torch.svd_lowrank(G, q=q, niter=4)
        else:
            U, S, V_transpose = torch.linalg.svd(G, full_matrices=False)
        total_energy = S.pow(2).sum()
        running = 0.0
        k = 0
        for i in range(len(S)):
            running += S[i].pow(2)
            if running/total_energy >= energy:
                k = i+1
                break

        V_transpose = V_transpose.T
        # print(f"{name}: V.shape = {V_transpose.shape}, k={k}")
        basis_per_layer[name] = V_transpose[:k, :].contiguous().clone().cpu()
    return basis_per_layer


def eval_and_save(args, model, val_preprocess, model_iteration_count, loss_dict):
    global gradient_per_layer_list
    if args.eval_datasets is not None:
        results = evaluate_2(model, args, val_preprocess)
    
        for dict in results:
            print(f"Saving {dict['dataset_name']} results")
            path = args.save + "/" + f"metrics_{dict['dataset_name']}.csv"

            rows=[]
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["iteration","top1","top5"])
                    writer.writeheader()

            with open(path, newline="") as f:
                reader=csv.DictReader(f)
                rows = list(reader)

            metrics = dict["metrics"]
            rows.append({
                    "iteration": model_iteration_count,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                })
            
            unique = {}
            for row in rows:
                unique[row["iteration"]] = row

            with open(path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["iteration","top1","top5"])
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(unique.values())
            print(f"Saving evaluation results to {path}...")

    del results
    torch.cuda.empty_cache()

def handle_signal(signum, frame):
    global gradients_per_layer
    print(f"Signaled end, {signum}\n Saving...")
    
    if model_ref is None:
        print("Model not initialized yet. Exitting...")
        sys.exit(0)

    saved_model = model_ref
    
    checkpoint = {
        "iteration": iteration_save,
        "state_dict": saved_model.state_dict(),
    }

    path = os.path.join(args_ref.save, f"{args_ref.train_dataset}.pth")
    #utils.torch_save(saved_model, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    
    print(f"Done saving: \nPath:{path}")

    print(f"Saving gradients...")
    if args_ref.orthogonal_gradients is not None:
        torch.save(gradients_per_layer, os.path.join(args_ref.save, f"grad_{args_ref.train_dataset}.pth"))
    print(f"Done saving: \nPath:{os.path.join(args_ref.save, f'grad_{args_ref.train_dataset}.pth')}")

    sys.exit(0)


signal.signal(signal.SIGUSR1, handle_signal)


def finetune(args):
    global model_ref, args_ref, iteration_save, gradient_save, gradients_per_layer
    prev_L2_loss = None
    prev_ZSCL_loss = None
    args_ref = args
    if args.untrained:
        model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, pretrained=True)
    else:
        model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, pretrained=False)

        
    model_iteration_count = 0
    if args.load is not None:
        utils.torch_load(model, args.load)
        
        model_iteration_count = torch.load(args.load)["iteration"]
        if args.start_iteration is not None:
            model_iteration_count = args.start_iteration
        print(f"Loaded checkpoint, total_iterations: {model_iteration_count}")

    if args.we_wise or (args.wise_merge and args.wise_ft_model != "zeroshot"):
        print("Using WiSE-FT with Loaded Model")
        model_fix, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
        if args.load is not None:
            utils.torch_load(model_fix, args.load)

    if args.we or args.moving_avg or args.we_wise:
        print("Averaging training")
        if args.moving_avg and args.mv_avg_model == "zeroshot": # mv+zeroshot
            we_model, _, _ =  clip.load(args.model, jit=False)
            we_model.cuda()
            we_n = 0
        else: #we; mv+m; mv+t; we_wise
            we_model = copy.deepcopy(model)
            we_model.cuda()
            we_n = 0
    if args.l2 > 0:
        print("L2 norm")
        l2_model = copy.deepcopy(model)
        l2_model.cuda()

    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    # prepare template
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    # number of iterations
    num_batches = len(dataset.train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval
    loss_interval = args.loss_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # get params
    if args.train_mode == "text":
        print("[Training mode] Text Encoder")
        visual_params_name = [k for k, v in model.visual.named_parameters()]
        exclude_params_name = visual_params_name + ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]
    elif args.train_mode == "image":
        print("[Training mode] Image Encoder")
        params = model.visual.parameters()
    else:
        assert args.train_mode == "whole"
        print("[Training mode] Both Encoders")
        exclude_params_name = ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]

    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    # Method
    if args.method == "ZSCL":
        # (Ref Model) get reference model
        print("[Method] ZSCL")
        if args.ref_model is None:
            if args.ref_wise:
                print("[ref_model] WiSE-Zero-shot")
                ref_model, _, test_preprocess = clip.load(args.model, jit=False)
                for param_q, param_k in zip(ref_model.parameters(), model.module.parameters()):
                    param_q.data = param_q.data * (1 - args.ref_wise_alpha) + param_k.data * args.ref_wise_alpha
            else:    
                print("[ref_model] Zero-shot")
                ref_model, _, test_preprocess = clip.load(args.model, jit=False)
        else:
            print(f"[ref_model] {args.ref_model}")
            ref_model, _, test_preprocess = clip.load(args.model, jit=False)
            utils.torch_load(
                ref_model, args.ref_model
            )
        ref_model = ref_model.cuda()
        ref_model = torch.nn.DataParallel(ref_model, device_ids=devices)
        ref_model.eval()

        # (Ref Dataset) get reference dataset
        ref_dataset_cls = getattr(datasets, args.ref_dataset)
        print(f"[Ref Dataset] {args.ref_dataset}")
        if args.ref_dataset in ["ImageNetSM", "ImageNetSUB"]:
            ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                num=args.num,
            )
        else:
            ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
            )
        ref_iter = iter(ref_dataset.train_loader)

        # (Ref Text) get reference text
        if args.text_datasets is not None:
            print("[Ref Sentences] Text-Datasets")
            ref_texts = get_datasets_text(args.text_datasets, args)
        elif args.ref_sentences == "random":
            ref_texts = virtual_vocab()
            print("[Ref Sentences] Random Sentences")
        elif args.ref_sentences is not None:
            ref_sentences_cls = getattr(datasets, args.ref_sentences)
            print(f"[Ref Sentences] {args.ref_sentences}")
            ref_sentences = ref_sentences_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
            )
            if args.ref_sentences == "conceptual_captions":
                # breakpoint()
                ref_texts = ref_sentences.train_dataset.captions
                ref_texts = clip.tokenize(ref_texts).cuda()

            else:
                ref_template = ref_sentences.template
                ref_texts = [ref_template(x) for x in ref_sentences.classnames]
                ref_texts = clip.tokenize(ref_texts).cuda()
        else:
            print(f"[Ref Sentences] {args.ref_dataset}")
            ref_template = ref_dataset.template
            ref_texts = [ref_template(x) for x in ref_dataset.classnames]
            ref_texts = clip.tokenize(ref_texts).cuda()
            
    if args.train_mode == "text":
        embeddings = zeroshot_classifier(dataset.classnames, dataset.templates, model)




    model_ref = model.module if hasattr(model, "module") else model
    
    mixup_image, mixup_label = None, None


    #orthogonal gradients
    if args.orthogonal_gradients is not None:
        # gradients_per_layer = defaultdict(list)
        for name, block in model.module.named_parameters():
            if should_track_layer(name):
                gradients_per_layer[name] = []
                block.register_hook(save_output_gradients(name))
        if args.save is not None:
            grad_path = os.path.join(args.save, f"grad_{args.train_dataset}.pth")
            if os.path.exists(grad_path):
                gradients_per_layer = torch.load(grad_path, weights_only=False)
                # for key in gradients_per_layer:
                #     print(f"{key}:{len(gradients_per_layer[key])}")
    
    #prev task
    if args.orthogonal_gradients_path is not None:    
        prev_basis = []
        for t in range(len(args.orthogonal_gradients_path)):
            basis = torch.load(args.orthogonal_gradients_path[t], weights_only=False)
            prev_basis.append(basis)


            
    for iteration in tqdm(range(model_iteration_count, total_iterations + 1)):
        gradient_save = False
        if args.orthogonal_gradients is not None:
            if iteration % total_iterations//args.orthogonal_gradients == 0:
                    gradient_save = True
        #saving references
        iteration_save = iteration


        if args.we or args.we_wise:
            model_ref = we_model
        args_ref = args

        

        if iteration % args.eval_interval == 0:
            print("Saving accuracies...")
            torch.cuda.empty_cache()

            loss_dict = {
                "ZSCL": prev_ZSCL_loss,
                "L2": prev_L2_loss
            }

            with torch.no_grad():
                eval_and_save(args, model, val_preprocess, iteration, loss_dict)
            torch.cuda.empty_cache()    

        # evaluation
        '''
        if eval_iterations is not None and iteration % eval_iterations == 0:
            if args.we or args.we_wise:
                evaluate(we_model, args, val_preprocess)
            else:
                evaluate(model.module, args, val_preprocess)
        '''
        # training
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        # prepare model
        model.train()
        scheduler(iteration)

        #freezing layers
        if args.freeze:
            freeze = [
                "positional_embedding",
                "visual.positional_embedding",
                "visual.class_embedding",
                "visual.proj",
                "visual.conv1.weight",
                "visual.ln_pre.weight",
                "visual.ln_pre.bias",
                "text_projection",
                "logit_scale",
            ]
            for name, param in model.module.named_parameters():
                if any(key in name for key in freeze):
                    param.requires_grad = False
        # prepare data
        if args.train_dataset == 'ImageNet':
            try:
                train_batch = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                train_batch = next(data_iter)
            images, labels = train_batch["images"], train_batch["labels"]
        else:
            try:
                images, labels = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()

        # ce loss
        # -- get text embedding --
        if args.train_mode != "text":
            
            embeddings = model(None, texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        # -- cross entropy loss --
        logits_per_image = logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

        
        '''
        if args.mixup is not None:
            beta_m = 0.5
            lamb = torch.Tensor([random.betavariate(beta_m,beta_m)]).to("cuda:0")
            theta = torch.acos( (logits_per_image*logits_per_text).sum(dim=[1])).view(logits_per_image.shape[0],1)
            n1 = torch.sin(lamb*theta)/torch.sin(theta)*logits_per_image
            n2 = torch.sin((1-lamb)*theta)/torch.sin(theta)*logits_per_text

            mixed = n1+n2
        '''



        if args.l2 > 0:
            loss_l2 = l2_loss(model, l2_model)
            loss += args.l2 * loss_l2

        if args.method == "ZSCL":

            # FIXME: only for ImageNet
            if args.ref_dataset in ["ImageNet", "ImageNetSM", "ImageNetSUB"]:
                try:
                    ref_batch = next(ref_iter)
                except:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_batch = next(ref_iter)
                ref_images, ref_labels = ref_batch["images"], ref_batch["labels"]
            else:
                try:
                    ref_images, ref_labels = next(ref_iter)
                except:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_images, ref_labels = next(ref_iter)
            ref_images, ref_labels = ref_images.cuda(), ref_labels.cuda()
            # breakpoint()
            
            
            with torch.no_grad():
                # -- get ref text embedding --
                
                ref_embeddings = ref_model(None, ref_texts)
                ref_embeddings = ref_embeddings / ref_embeddings.norm(
                    dim=-1, keepdim=True
                )

                # -- get ref image embedding --
                ref_out = ref_model(ref_images, None)
                ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)

            # -- get image embedding --
            ref_out_current = model(ref_images, None)
            ref_out_current = ref_out_current / ref_out_current.norm(
                dim=-1, keepdim=True
            )

            # -- loss --
            logits_current = logit_scale.exp() * ref_out_current @ ref_embeddings.t()
            logits_ref = logit_scale.exp() * ref_out @ ref_embeddings.t()
            loss_ZSCL = distillation(logits_ref, logits_current, T=args.T)

            # feature-space mse
            if args.feature_mse:
                mse_loss = torch.nn.MSELoss()
                loss += mse_loss(ref_out, ref_out_current)
            # -- final loss --
            if args.image_loss:
                if args.weight_adjust:
                    loss = loss + 0.5 * loss_ZSCL 
                else:
                    loss = loss + 1.0 * loss_ZSCL 

            # transpose loss
            if args.text_loss:
                logits_current_2 = logits_current.t()
                logits_ref_2 = logits_ref.t()
                loss_ZSCL_2 = distillation(logits_ref_2, logits_current_2, T=args.T)
                if args.weight_adjust:
                    loss += 0.5 * loss_ZSCL_2
                else:
                    loss += loss_ZSCL_2
            
            if args.ablation_loss_2:
                logits_img_current = logit_scale.exp() * ref_out_current @ ref_out_current.t()
                logits_img_ref = logit_scale.exp() * ref_out @ ref_out.t()
                logits_img_current -= torch.diag(logits_img_current.diag() + 1e4)
                logits_img_ref -= torch.diag(logits_img_ref.diag() + 1e4)
                loss_ZSCL_3 = distillation(logits_img_ref, logits_img_current, T=args.T)
                if args.weight_adjust:
                    loss += 0.5 * loss_ZSCL_3
                else:
                    loss += loss_ZSCL_3


        # update
        optimizer.zero_grad()
        loss.backward()

        #OGD, modify gradients before stepping
        if args.orthogonal_gradients_path is not None:
            new_gradients = {}
            for name, gradient in gradients_per_layer.items():
                # print(f"{name}:{gradient}")
                gradient = gradient[-1].to("cuda:0")
                for basis_dict in prev_basis:
                    #prev layers does not match with current layers
                    if name not in basis_dict:
                        continue
                    B = basis_dict[name]
                    # if B.shape[0] != gradient.numel():    
                    #     B = B.T

                    # if isinstance(B, list):
                    #     B = torch.stack(B, dim=0).contiguous().clone().cpu()
                    B = B.to(gradient.device)
                    
                    proj_g_ort_B = B.T @ (B @ gradient)
                    gradient = gradient - proj_g_ort_B

                new_gradients[name] = gradient

            param_dict = dict(model.named_parameters())
            for name, gradient in new_gradients.items():
                if name not in param_dict:
                    continue
                param = param_dict[name]
                param.grad = gradient.view_as(param)
        optimizer.step()

        # we
        if (args.we or args.moving_avg or args.we_wise) and iteration % args.avg_freq == 0:
            we_n += 1
            if args.moving_avg:
                if args.mv_avg_model == "t":
                    next_we_model = copy.deepcopy(model.module)
                    moving_avg(model.module, we_model, args.mv_avg_decay)
                    we_model = next_we_model.cuda()
                else: ### args.moving_avg_model == "n" or "zeroshot"
                    moving_avg(model.module, we_model, args.mv_avg_decay)
            elif args.we:
                merge_we(model.module, we_model, we_n)
            else:
                wise_we(model.module, we_model, we_n, model_fix, args.we_wise_alpha)
        
        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())
            if args.method == "ZSCL":
                prev_ZSCL_loss = loss_ZSCL.item() 
                print("Loss ZSCL:", prev_ZSCL_loss)
            if args.l2 > 0:
                prev_L2_loss = loss_l2.item() 
                print("Loss L2:", loss_l2.item())

        # if args.orthogonal_gradients is not None:
        #     if iteration % total_iterations//args.orthogonal_gradients == 0:


                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         continue
                    
                #     gradient = param.grad.detach().clone().flatten()

                #     gradients_per_layer[name].append(gradient)

        
        # mixup_image, mixup_label = images.copy(), labels.copy()

        #
    if args.wise_merge:
        alpha = args.wise_ft_alpha
        if args.wise_ft_model == "zeroshot":
            wise_ft_model, _, _ =  clip.load(args.model, jit=False)
        else:
            wise_ft_model = copy.deepcopy(model_fix)
        wise_ft_model.cuda()
        for param_q, param_k in zip(model.module.parameters(), wise_ft_model.parameters()):
            param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)

    #Saving gradients
    if args.orthogonal_gradients:
        basis_per_layer = svd_basis(gradients_per_layer)
        torch.save(basis_per_layer, os.path.join(args.save, f"grad_{args.train_dataset}.pth"))

    # Saving model
    if args.save is not None:
        if args.we or args.we_wise:
            to_save_model = we_model
        else:
            to_save_model = model.module
        # to_save_model = model.module
        #utils.torch_save(to_save_model, path)
        
        checkpoint = {
            "iteration": iteration_save,
            "state_dict": to_save_model.state_dict(),
        }

        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
