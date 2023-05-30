import torch
import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding2

class BERT5(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    
    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
            
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken,std=0.02)
        
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding2(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])


    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample