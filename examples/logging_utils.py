import wandb


def wandb_log_curve(xs, ys, x_name, y_name, title, id):
    data = [[x, y] for (x, y) in zip(xs, ys)]
    table = wandb.Table(data=data, columns=[x_name, y_name])
    wandb.log({id: wandb.plot.line(table, x_name, y_name, title=title)})
