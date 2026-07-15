import torch

def targeted_attack_on_reflectance(net, criterion, ref_img, targets,
                                   eps=4/255., eps_iter=0.5/255., n_iter=10):
    """
    对单张反射图进行目标攻击，生成伪真值
    参数：
        net       : DSFD 检测器（可能被 DDP 包装）
        criterion : MultiBoxLoss 实例
        ref_img   : 反射图张量 [1, 3, H, W]，0~1
        targets   : 该图对应的目标列表（格式同训练）
        eps, eps_iter, n_iter : 攻击超参数
    返回：
        adv_img   : 扰动后的伪真值 [1,3,H,W]
    """
    model = net.module if hasattr(net, 'module') else net
    adv_img = ref_img.clone().detach().requires_grad_(True)

    # 动量参数
    momentum = 0
    v = 0
    beta1, beta2 = 0.9, 0.999
    eps = torch.tensor(eps, device=ref_img.device)
    eps_iter = torch.tensor(eps_iter, device=ref_img.device)

    for t in range(1, n_iter + 1):
        # 前向检测
        out = model.forward_detection(adv_img)
        # 计算检测损失（分类 + 回归）
        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2

        # 梯度 wrt adv_img
        grad = torch.autograd.grad(loss, adv_img, retain_graph=False)[0]

        # Adam 风格动量更新
        momentum = beta1 * momentum + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = momentum / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (torch.sqrt(v_hat) + 1e-8)

        # 梯度下降（减号，最小化损失）
        adv_img = adv_img - eps_iter * update.sign()

        # 裁剪扰动至 [−eps, eps] 并限制像素范围
        with torch.no_grad():
            eta = torch.clamp(adv_img - ref_img, -eps, eps)
            adv_img = torch.clamp(ref_img + eta, 0.0, 1.0)
            adv_img = adv_img.detach().requires_grad_(True)

    return adv_img.detach()

