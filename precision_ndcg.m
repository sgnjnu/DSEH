function [map acg ndcg] = precision_ndcg(trn_label, trn_binary, tst_label, tst_binary, top_k, mode)   
K = top_k;


QueryTimes = size(tst_binary,1)
% iiii=randperm(size(tst_label,1));
% tst_binary=tst_binary(iiii,:);
% tst_label=tst_label(iiii,:);
% QueryTimes=100;
correct = zeros(K,1);
total = zeros(K,1);
error = zeros(K,1);
AP = zeros(QueryTimes,1);
ACG=zeros(QueryTimes,1);
NDCG=zeros(QueryTimes,1);
Ns = 1:1:K;
sum_tp = zeros(1, length(Ns));

for i = 1:QueryTimes
    
    query_label = tst_label(i,:);
    fprintf('query %d\n',i);
    query_binary = tst_binary(i,:);
    if mode==1
    tic
    similarity = pdist2(trn_binary,query_binary,'hamming');
    toc
    fprintf('Complete Query [Hamming] %.2f seconds\n',toc);
    elseif mode==3
    tic
    similarity=zeros(size(trn_binary,1),1);
      for ii=1:length(similarity)
         similarity(ii)=sum(bitxor(trn_binary(ii,:),query_binary))/length(query_binary);
      end
%      similarity=compute_hamming_distance(query_binary',trn_binary');
    toc
    fprintf('Complete Query [Hamming] %.2f seconds\n',toc);
    elseif mode ==2
    tic
    similarity = pdist2(trn_binary,query_binary,'euclidean');
    toc
    fprintf('Complete Query [Euclidean] %.2f seconds\n',toc);
   elseif mode ==4
    tic
    similarity = pdist2(trn_binary,query_binary,'cosine');
    toc
    fprintf('Complete Query [Euclidean] %.2f seconds\n',toc);
    end

    [x2,y2]=sort(similarity);
    
    buffer_yes = zeros(K,1);
    buffer_total = zeros(K,1);
    total_relevant = 0;
    acg=zeros(K,1);
    for j = 1:K
        retrieval_label = trn_label(y2(j),:);
        
        if (sum(query_label&retrieval_label)~=0)
            buffer_yes(j,1) = 1;
            total_relevant = total_relevant + 1;
            acg(j)=sum(query_label&retrieval_label);
        end
        buffer_total(j,1) = 1;
    end
    
    ACG(i)=sum(acg)/K;
    % compute  weight precision
    %P = cumsum(buffer_yes) ./ Ns';
    Acgs= cumsum(acg)./ Ns';
    Dcgs=(2.^acg-1)./log2(Ns'+1);
%     idcg=iDCG(acg);
    idcg=iDCG_full(query_label,trn_label,K);
    if idcg==0
        NDCG(i)=0;
    else
    NDCG(i)=sum(Dcgs)/idcg;
    end
    
   if (sum(buffer_yes) == 0)
       AP(i) = 0;
   else
       AP(i) = sum(Acgs.*buffer_yes) / sum(buffer_yes);
   end    
end  
acg=mean(ACG)
map = mean(AP);
ndcg=mean(NDCG);
end