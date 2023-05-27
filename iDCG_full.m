function idcg=iDCG_full(query_label,train_labels,K1)
scores=query_label*(train_labels');
[perfect_rank,~]=sort(scores,'descend');
perfect_rank=perfect_rank(1:K1);
K=1:K1;
idcg=sum((2.^perfect_rank-1)./log2(K+1));
end