import torch
import pdb
import torch.nn.functional as F

class Cross_entropy(torch.nn.Module):
    def __init__(self):
        super(Cross_entropy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
    def forward(self, output, label, pos_neg_mask=None):
        # output = output.view(-1, output.shape[-1])
        # label = label.view(-1)
        # pdb.set_trace()
        loss = self.loss(output, label)*(1-pos_neg_mask)

        acc = (output.argmax(dim=1) == label).float()
        acc = acc * (1-pos_neg_mask)
        mask_sum = (1 - pos_neg_mask).sum()
        acc = acc.sum() / mask_sum if mask_sum > 0 else torch.tensor(0.0, device=output.device)

        # pdb.set_trace()

        return (loss.mean(), acc)

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive

def compute_map(query_feats, gallery_feats, query_labels, gallery_labels):
    """
    Compute mean average precision (mAP) for the given query and gallery features and labels.
    """
    # Normalize the features
    query_feats = F.normalize(query_feats, dim=1) # 4, 768
    gallery_feats = F.normalize(gallery_feats, dim=1) # 4, 768
    # pdb.set_trace()

    # Compute cosine similarity
    similarity_matrix = torch.mm(query_feats, gallery_feats.t()) # 4,4

    # Get the indices of the sorted similarities
    _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True) # 4,4

    # Initialize variables to compute mAP
    num_queries = query_feats.size(0)
    average_precisions = []

    for i in range(num_queries): # 4
        query_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[sorted_indices[i]] #4

        # Compute precision at each rank
        correct_matches = (sorted_gallery_labels == query_label).float() #4
        precision_at_k = correct_matches.cumsum(0) / (torch.arange(len(correct_matches)) + 1).float() #4

        # Compute average precision
        average_precision = (precision_at_k * correct_matches).sum() / correct_matches.sum()
        average_precisions.append(average_precision.item())

    # Compute mean average precision
    map_value = sum(average_precisions) / num_queries 

    return map_value

if __name__ == "__main__":
    ar1 = torch.randn(4, 768)
    ar2 = torch.randn(4, 768)
    label = torch.tensor([0, 1, 0, 1])
    out = compute_map(ar1, ar2, label, label)
