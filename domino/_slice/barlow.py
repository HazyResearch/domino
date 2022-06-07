



def failure_explanation(imagenet_path, class_name, grouping, model_name = "standard"):
    robust_model_name = 'robust_resnet50.pth'
    robust_model = load_robust_model()
    
    imagenet_subset = ImageNetSubset(imagenet_path, class_name, grouping, model_name)
        
    train_features, train_labels, train_preds = extract_features(robust_model, imagenet_subset)
    
    train_success = (train_preds == train_labels)
    train_failure = np.logical_not(train_success)
    
    train_base_error_rate = np.sum(train_failure)/len(train_failure)
    
    sparse_features, feature_indices = select_important_features(train_features, train_failure, 
                                                                 num_features=50, method='mutual_info')    
    
    decision_tree = train_decision_tree(sparse_features, train_failure, 
                                        max_depth=1, criterion="entropy")
    train_precision, train_recall, train_ALER = decision_tree.compute_precision_recall(
        sparse_features, train_failure)
    
    class_name = class_names[class_index]
    
    
    
    print_with_stars(" Training Data Summary ", prefix="\n")
    print('Grouping by {:s} for class name: {:s}'.format(grouping, class_name))
    print('Number of correctly classified: {:d}'.format(np.sum(train_success)))
    print('Number of incorrectly classified: {:d}'.format(np.sum(train_failure)))
    print('Total size of the dataset: {:d}'.format(len(train_failure)))
    print('Train Base_Error_Rate (BER): {:.4f}\n'.format(train_base_error_rate))

    print_with_stars(" Decision Tree Summary (evaluated on training data) ")
    print('Tree Precision: {:.4f}'.format(train_precision))
    print('Tree Recall: {:.4f}'.format(train_recall))
    print('Tree ALER (ALER of the root node): {:.4f}\n'.format(train_ALER))

    
    error_rate_array, error_coverage_array = decision_tree.compute_leaf_error_rate_coverage(
                                                sparse_features, train_failure)

    important_leaf_ids = important_leaf_nodes(decision_tree, error_rate_array, error_coverage_array)
    for leaf_id in important_leaf_ids[:1]:
        leaf_precision = error_rate_array[leaf_id]
        leaf_recall = error_coverage_array[leaf_id]

        decision_path = decision_tree.compute_decision_path(leaf_id)

        print_with_stars(" Failure statistics for leaf[{:d}] ".format(leaf_id))
        print('Leaf Error_Rate (ER): {:.4f}'.format(leaf_precision))
        print('Leaf Error_Coverage (EC): {:.4f}'.format(leaf_recall))
        print('Leaf Importance_Value (IV): {:.4f}'.format(leaf_precision*leaf_recall))

        
        leaf_failure_indices = decision_tree.compute_leaf_truedata(sparse_features, 
                                                                   train_failure, leaf_id)
        display_failures(leaf_id, leaf_failure_indices, imagenet_subset, grouping, num_images=6)
        
        print_with_stars(" Decision tree path from root to leaf[{:d}] ".format(leaf_id))
        for node in decision_path:
            node_id, feature_id, feature_threshold, direction = node
            
            if direction == 'left':
                print_str = "Feature[{:d}] < {:.6f} (left branching, lower feature activation)".format(
                    feature_id, feature_threshold)
            else:
                print_str = "Feature[{:d}] > {:.6f} (right branching, higher feature activation)".format(
                    feature_id, feature_threshold)

            print(print_str)
        print("")
        
        print_with_stars(" Visualizing features on path from root to leaf[{:d}] ".format(leaf_id))
        print("")
        display_images(decision_path, imagenet_subset, robust_model, train_features, 
                       feature_indices, grouping, num_images=6)