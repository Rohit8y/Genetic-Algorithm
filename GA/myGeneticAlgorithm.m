
% Genetic algorithm for the selection of the best subset of feature

function bestchromosome = myGeneticAlgorithm(features,labels)
% features = A matrix of independent variables
% labels = A vector that contains the labels for each rows in matrix features


nchroms       = 100; % number of chromosomes (population size)
nepochs       = 10;  % number of epochs (number of generations)
nparentsratio = 0.2; % portion of elite list (elite ratio)
mutateprob    = 0.1; % probability to mutate a bit in a chromosome

% Create figure that shows the progress of the genetic algorithm
figure;hold on;
title('Feature Selection with Genetic Algorithm');
colorlist = jet(nepochs);

% Convert labels, which can be in string format, to numeric.
[lbls,h] = grp2idx(labels);

% Iterate through all epochs
for epoch = 1:nepochs
    fprintf('epoch %d of %d\n',epoch,nepochs);
    if epoch == 1
        % generate the intial popultion of chromosome with randomly
        % assigned bits
        pop = generateInitialPopulation(nchroms,size(features,2));        
    else
        % generate a new population by creating offspring from the best
        % performing chromosome (or parents)
        pop = getnewpopulation(pop,score,nparentsratio,mutateprob);
    end
    
    pop = logical(pop);    
    % Compute the fitness score for each chromosome
    score = zeros(1,nchroms);
    for i = 1:nchroms
        score(i) = getScore(pop(i,:),features,lbls);    
    end    
    
    % Plot the scores to visualize the progress
    plot(sort(score,'descend'),'color',colorlist(epoch,:));
    xlabel('Chromosome');
    ylabel('Fitness Score');
    legendList{epoch} = sprintf('Epoch %d',epoch);
    legend(legendList);
    drawnow;
end

% Return the chromosome with the maximum fitness score
[~,mxind] = max(score);
bestchromosome = pop(mxind,:);

function newpop = getnewpopulation(pop,score,nparentsratio,mutateprob)
% Generate a new population by first selecting the best performing
% chromosomes from the given pop matix, and subsequently generate new offspring chromosomes from randomly
% selected pairs of parent chromosomes.

% Step 1. Write code to select the top performing chromosomes. Use nparentsratio to
% calculate how many parents you need. If pop has 100 rows and
% nparentsration is 0.2, then you have to select the top performing 20
% chromosomes
[~,ind] = sort(score,'descend');
nparents = nparentsratio * size(pop,1);

newpop = zeros(size(pop));
newpop(1:nparents,:) = pop(ind(1:nparents),:);

topparents = pop(ind(1:nparents),:);

% Step 2. Iterate until a new population of the same size is generated. Using the above
% example, you need to iterate 80 times. In each iteration create a new
% offspring chromosome from two randomly selected parent chromosomes. Use
% the function getOffSpring to generate a new offspring.

for j = nparents+1:size(pop,1)    
    randparents = randperm(nparents);    
    newpop(j,:) = getOffSpring(topparents(randparents(1),:),topparents(randparents(2),:),mutateprob,'double');    
end

function offspring = getOffSpring(parent1,parent2,mutateprob,type)
% Generate an offspring from parent1 and parent2 and mutate the bits by
% using the probability mutateprob.
% TO FILL IN

% Step 1. Write code that generates one offspring from the given two parents

% Find first random point between 1 and size of features of parent
index1 = randi(size(parent1,2)); 

% Find second random point between 1 and size of features of parent making
% sure that the second random point is not same as first random point
x = setdiff(1:size(parent1,2), [index1]);
index2 = x(randi(numel(x)));

% If the type of crossover is single then we will only use one point to
% divide both the parents on the basis of index1 and then select one
% offspring
if strcmp(type,'single') == 1
    offspring = [parent1(1:index1) parent2(index1+1:end)];
end

% if the crossover is 2-point then we will split both the parents on the
% basis of index1 and index2 and generate one offspring, although we want
% to make sure that index2 is greater than index1.
if strcmp(type,'double') == 1
    if index1<index2
        offspring = [parent1(1:index1) parent2(index1+1:index2) parent1(index2+1:end)];
    end
    if index2<index1
        offspring = [parent1(1:index2) parent2(index2+1:index1) parent1(index1+1:end)];
    end  
end

% Step 2. Write code to mutate some bits with given mutation probability mutateprob

for i = 1: size(offspring,2)
    random = rand();
    oldValue=offspring(i);
    if random < mutateprob
        offspring(i) = ~ oldValue;
    end
end

function score = getScore(chromosome,train_feats,labels)
% Compute the fitness score using 2-fold cross validation and KNN
% classifier

cv = cvpartition(labels,'Kfold',2);
for i = 1:cv.NumTestSets        
    knn = fitcknn(train_feats(cv.training(i),chromosome),labels(cv.training(i)));
    c = predict(knn,train_feats(cv.test(i),chromosome));
    acc(i) = sum(c == labels(cv.test(i)))/numel(c);
end
meanacc = mean(acc);


% TO FILL IN: WRITE CODE TO CALCULATE THE SCORE FOR THE GIVEN CHROMOSOME
% BASED ON THE TRAINING ACCURACY MEANACC AND THE NUMBER OF ZEROS IN THE
% CHROMOSOME

% count the number of zero in chromosome
n=chromosome==0;
count=sum(n(:));

%calculate the score
score=(10000*meanacc)+(0.4*count);

function pop = generateInitialPopulation(n,ndim)
% TO FILL IN: Generate the initial population of chromosomes with random bits
pop = randi([0,1],n,ndim);





