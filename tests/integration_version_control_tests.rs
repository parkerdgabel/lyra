use lyra::vm::{Value, VM};
use lyra::stdlib::StandardLibrary;
use lyra::foreign::{LyObj, Foreign, ForeignError};
use std::collections::HashMap;

/// Comprehensive tests for version control integration functions
/// These tests verify Git integration, repository management, and code review automation

#[cfg(test)]
mod version_control_tests {
    use super::*;

    fn create_vm_with_integrations() -> VM {
        let mut vm = VM::new();
        let mut stdlib = StandardLibrary::new();
        stdlib.register_integration_functions();
        for (name, func) in stdlib.function_names().iter().zip(stdlib.function_names().iter()) {
            vm.register_function(name, *stdlib.get_function(name).unwrap());
        }
        vm
    }

    #[test]
    fn test_git_repository_creation() {
        let vm = create_vm_with_integrations();
        
        // Test GitRepository[path, remote_url, credentials]
        let result = vm.eval("GitRepository[\"/tmp/test-repo\", \"https://github.com/user/repo.git\", {\"token\" -> \"ghp_test\"}]");
        
        assert!(result.is_ok());
        let repo = result.unwrap();
        
        // Should return a GitRepository Foreign object
        match repo {
            Value::LyObj(obj) => {
                assert_eq!(obj.type_name(), "GitRepository");
                
                // Test repository methods
                let path_result = obj.call_method("getPath", &[]);
                assert!(path_result.is_ok());
                assert_eq!(path_result.unwrap(), Value::String("/tmp/test-repo".to_string()));
                
                let remote_result = obj.call_method("getRemoteUrl", &[]);
                assert!(remote_result.is_ok());
                assert_eq!(remote_result.unwrap(), Value::String("https://github.com/user/repo.git".to_string()));
                
                let status_result = obj.call_method("getStatus", &[]);
                assert!(status_result.is_ok());
            }
            _ => panic!("Expected GitRepository LyObj"),
        }
    }

    #[test]
    fn test_git_repository_invalid_args() {
        let vm = create_vm_with_integrations();
        
        // Test with missing arguments
        let result = vm.eval("GitRepository[\"/tmp/test-repo\"]");
        assert!(result.is_err());
        
        // Test with invalid path type
        let result = vm.eval("GitRepository[123, \"https://github.com/user/repo.git\", {}]");
        assert!(result.is_err());
    }

    #[test]
    fn test_git_commit_creation() {
        let vm = create_vm_with_integrations();
        
        // First create a repository
        let repo_result = vm.eval("GitRepository[\"/tmp/test-repo\", \"https://github.com/user/repo.git\", {}]");
        assert!(repo_result.is_ok());
        
        // Test GitCommit[repo, message, files, author]
        let result = vm.eval("GitCommit[repo, \"Add new feature\", {\"src/feature.rs\", \"tests/feature_test.rs\"}, {\"name\" -> \"Developer\", \"email\" -> \"dev@example.com\"}]");
        
        assert!(result.is_ok());
        let commit = result.unwrap();
        
        match commit {
            Value::LyObj(obj) => {
                assert_eq!(obj.type_name(), "GitCommit");
                
                // Test commit methods
                let hash_result = obj.call_method("getHash", &[]);
                assert!(hash_result.is_ok());
                
                let message_result = obj.call_method("getMessage", &[]);
                assert!(message_result.is_ok());
                assert_eq!(message_result.unwrap(), Value::String("Add new feature".to_string()));
                
                let files_result = obj.call_method("getFiles", &[]);
                assert!(files_result.is_ok());
                
                let author_result = obj.call_method("getAuthor", &[]);
                assert!(author_result.is_ok());
            }
            _ => panic!("Expected GitCommit LyObj"),
        }
    }

    #[test]
    fn test_git_branch_operations() {
        let vm = create_vm_with_integrations();
        
        // Create repository first
        let repo_result = vm.eval("GitRepository[\"/tmp/test-repo\", \"https://github.com/user/repo.git\", {}]");
        assert!(repo_result.is_ok());
        
        // Test GitBranch[repo, branch_name, base_branch]
        let result = vm.eval("GitBranch[repo, \"feature-branch\", \"main\"]");
        
        assert!(result.is_ok());
        let branch = result.unwrap();
        
        match branch {
            Value::LyObj(obj) => {
                assert_eq!(obj.type_name(), "GitBranch");
                
                // Test branch methods
                let name_result = obj.call_method("getName", &[]);
                assert!(name_result.is_ok());
                assert_eq!(name_result.unwrap(), Value::String("feature-branch".to_string()));
                
                let base_result = obj.call_method("getBaseBranch", &[]);
                assert!(base_result.is_ok());
                assert_eq!(base_result.unwrap(), Value::String("main".to_string()));
                
                let commits_result = obj.call_method("getCommits", &[]);
                assert!(commits_result.is_ok());
                
                // Test branch operations
                let checkout_result = obj.call_method("checkout", &[]);
                assert!(checkout_result.is_ok());
                
                let push_result = obj.call_method("push", &[]);
                assert!(push_result.is_ok());
            }
            _ => panic!("Expected GitBranch LyObj"),
        }
    }

    #[test]
    fn test_git_merge_operations() {
        let vm = create_vm_with_integrations();
        
        // Test GitMerge[repo, source_branch, target_branch, strategy]
        let result = vm.eval("GitMerge[repo, \"feature-branch\", \"main\", \"merge\"]");
        
        assert!(result.is_ok());
        let merge = result.unwrap();
        
        match merge {
            Value::LyObj(obj) => {
                assert_eq!(obj.type_name(), "GitMerge");
                
                // Test merge methods
                let source_result = obj.call_method("getSourceBranch", &[]);
                assert!(source_result.is_ok());
                
                let target_result = obj.call_method("getTargetBranch", &[]);
                assert!(target_result.is_ok());
                
                let strategy_result = obj.call_method("getStrategy", &[]);
                assert!(strategy_result.is_ok());
                assert_eq!(strategy_result.unwrap(), Value::String("merge".to_string()));
                
                let conflicts_result = obj.call_method("getConflicts", &[]);
                assert!(conflicts_result.is_ok());
                
                // Test merge execution
                let execute_result = obj.call_method("execute", &[]);
                assert!(execute_result.is_ok());
            }
            _ => panic!("Expected GitMerge LyObj"),
        }
    }

    #[test]
    fn test_commit_analysis() {
        let vm = create_vm_with_integrations();
        
        // Test CommitAnalysis[repo, commit_range, metrics]
        let result = vm.eval("CommitAnalysis[repo, \"v1.0..HEAD\", {\"lines_changed\", \"files_modified\", \"complexity\"}]");
        
        assert!(result.is_ok());
        let analysis = result.unwrap();
        
        match analysis {
            Value::LyObj(obj) => {
                assert_eq!(obj.type_name(), "CommitAnalysis");
                
                // Test analysis methods
                let range_result = obj.call_method("getCommitRange", &[]);
                assert!(range_result.is_ok());
                assert_eq!(range_result.unwrap(), Value::String("v1.0..HEAD".to_string()));
                
                let metrics_result = obj.call_method("getMetrics", &[]);
                assert!(metrics_result.is_ok());
                
                let lines_changed_result = obj.call_method("getLinesChanged", &[]);
                assert!(lines_changed_result.is_ok());
                
                let files_modified_result = obj.call_method("getFilesModified", &[]);
                assert!(files_modified_result.is_ok());
                
                let complexity_result = obj.call_method("getComplexity", &[]);
                assert!(complexity_result.is_ok());
                
                let authors_result = obj.call_method("getAuthors", &[]);
                assert!(authors_result.is_ok());
                
                let timeline_result = obj.call_method("getTimeline", &[]);
                assert!(timeline_result.is_ok());
            }
            _ => panic!("Expected CommitAnalysis LyObj"),
        }
    }

    #[test]
    fn test_branch_strategy() {
        let vm = create_vm_with_integrations();
        
        // Test BranchStrategy[repo, strategy_type, rules]
        let result = vm.eval("BranchStrategy[repo, \"gitflow\", {\"main_branch\" -> \"main\", \"develop_branch\" -> \"develop\", \"feature_prefix\" -> \"feature/\"}]");
        
        assert!(result.is_ok());
        let strategy = result.unwrap();
        
        match strategy {
            Value::LyObj(obj) => {
                assert_eq!(obj.type_name(), "BranchStrategy");
                
                // Test strategy methods
                let type_result = obj.call_method("getStrategyType", &[]);
                assert!(type_result.is_ok());
                assert_eq!(type_result.unwrap(), Value::String("gitflow".to_string()));
                
                let rules_result = obj.call_method("getRules", &[]);
                assert!(rules_result.is_ok());
                
                let main_branch_result = obj.call_method("getMainBranch", &[]);
                assert!(main_branch_result.is_ok());
                
                let develop_branch_result = obj.call_method("getDevelopBranch", &[]);
                assert!(develop_branch_result.is_ok());
                
                // Test strategy operations
                let create_feature_result = obj.call_method("createFeatureBranch", &[Value::String("new-feature".to_string())]);
                assert!(create_feature_result.is_ok());
                
                let create_release_result = obj.call_method("createReleaseBranch", &[Value::String("v1.2.0".to_string())]);
                assert!(create_release_result.is_ok());
            }
            _ => panic!("Expected BranchStrategy LyObj"),
        }
    }

    #[test]
    fn test_git_hooks() {
        let vm = create_vm_with_integrations();
        
        // Test GitHooks[repo, hook_type, script, triggers]
        let result = vm.eval("GitHooks[repo, \"pre-commit\", \"#!/bin/bash\\necho 'Running pre-commit hook'\", {\"lint\", \"format\", \"test\"}]");
        
        assert!(result.is_ok());
        let hooks = result.unwrap();
        
        match hooks {
            Value::LyObj(obj) => {
                assert_eq!(obj.type_name(), "GitHooks");
                
                // Test hooks methods
                let type_result = obj.call_method("getHookType", &[]);
                assert!(type_result.is_ok());
                assert_eq!(type_result.unwrap(), Value::String("pre-commit".to_string()));
                
                let script_result = obj.call_method("getScript", &[]);
                assert!(script_result.is_ok());
                
                let triggers_result = obj.call_method("getTriggers", &[]);
                assert!(triggers_result.is_ok());
                
                // Test hook operations
                let install_result = obj.call_method("install", &[]);
                assert!(install_result.is_ok());
                
                let execute_result = obj.call_method("execute", &[]);
                assert!(execute_result.is_ok());
                
                let validate_result = obj.call_method("validate", &[]);
                assert!(validate_result.is_ok());
            }
            _ => panic!("Expected GitHooks LyObj"),
        }
    }

    #[test]
    fn test_code_review() {
        let vm = create_vm_with_integrations();
        
        // Test CodeReview[repo, pull_request, reviewers, criteria]
        let result = vm.eval("CodeReview[repo, 123, {\"reviewer1\", \"reviewer2\"}, {\"code_quality\", \"test_coverage\", \"documentation\"}]");
        
        assert!(result.is_ok());
        let review = result.unwrap();
        
        match review {
            Value::LyObj(obj) => {
                assert_eq!(obj.type_name(), "CodeReview");
                
                // Test review methods
                let pr_result = obj.call_method("getPullRequestId", &[]);
                assert!(pr_result.is_ok());
                assert_eq!(pr_result.unwrap(), Value::Integer(123));
                
                let reviewers_result = obj.call_method("getReviewers", &[]);
                assert!(reviewers_result.is_ok());
                
                let criteria_result = obj.call_method("getCriteria", &[]);
                assert!(criteria_result.is_ok());
                
                let status_result = obj.call_method("getStatus", &[]);
                assert!(status_result.is_ok());
                
                // Test review operations
                let request_review_result = obj.call_method("requestReview", &[]);
                assert!(request_review_result.is_ok());
                
                let auto_review_result = obj.call_method("performAutoReview", &[]);
                assert!(auto_review_result.is_ok());
                
                let approve_result = obj.call_method("approve", &[Value::String("reviewer1".to_string())]);
                assert!(approve_result.is_ok());
                
                let get_comments_result = obj.call_method("getComments", &[]);
                assert!(get_comments_result.is_ok());
            }
            _ => panic!("Expected CodeReview LyObj"),
        }
    }

    #[test]
    fn test_version_control_integration() {
        let vm = create_vm_with_integrations();
        
        // Test complete workflow integration
        let workflow = "
            repo = GitRepository[\"/tmp/integration-test\", \"https://github.com/test/repo.git\", {}];
            branch = GitBranch[repo, \"feature-integration\", \"main\"];
            commit = GitCommit[repo, \"Integration test commit\", {\"src/integration.rs\"}, {\"name\" -> \"Test\", \"email\" -> \"test@example.com\"}];
            analysis = CommitAnalysis[repo, \"HEAD~1..HEAD\", {\"lines_changed\"}];
            strategy = BranchStrategy[repo, \"simple\", {\"main_branch\" -> \"main\"}];
            hooks = GitHooks[repo, \"pre-push\", \"exit 0\", {\"validate\"}];
            review = CodeReview[repo, 1, {\"auto-reviewer\"}, {\"basic_checks\"}];
            {repo, branch, commit, analysis, strategy, hooks, review}
        ";
        
        let result = vm.eval(workflow);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::List(objects) => {
                assert_eq!(objects.len(), 7);
                
                // Verify each object type
                for (i, obj) in objects.iter().enumerate() {
                    match obj {
                        Value::LyObj(lyobj) => {
                            let expected_types = ["GitRepository", "GitBranch", "GitCommit", "CommitAnalysis", "BranchStrategy", "GitHooks", "CodeReview"];
                            assert_eq!(lyobj.type_name(), expected_types[i]);
                        }
                        _ => panic!("Expected LyObj at position {}", i),
                    }
                }
            }
            _ => panic!("Expected list of integration objects"),
        }
    }

    #[test]
    fn test_error_handling() {
        let vm = create_vm_with_integrations();
        
        // Test invalid repository path
        let result = vm.eval("GitRepository[\"/nonexistent/path\", \"invalid-url\", {}]");
        assert!(result.is_err());
        
        // Test commit with invalid repository
        let result = vm.eval("GitCommit[Invalid[], \"message\", {}, {}]");
        assert!(result.is_err());
        
        // Test merge with conflicting branches
        let result = vm.eval("GitMerge[repo, \"nonexistent-branch\", \"main\", \"invalid-strategy\"]");
        assert!(result.is_err());
        
        // Test analysis with invalid commit range
        let result = vm.eval("CommitAnalysis[repo, \"invalid..range\", {}]");
        assert!(result.is_err());
    }

    #[test]
    fn test_concurrent_operations() {
        let vm = create_vm_with_integrations();
        
        // Test thread safety of version control operations
        let concurrent_ops = "
            repo = GitRepository[\"/tmp/concurrent-test\", \"https://github.com/test/repo.git\", {}];
            Parallel[{
                GitBranch[repo, \"branch1\", \"main\"],
                GitBranch[repo, \"branch2\", \"main\"],
                GitBranch[repo, \"branch3\", \"main\"]
            }]
        ";
        
        let result = vm.eval(concurrent_ops);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::List(branches) => {
                assert_eq!(branches.len(), 3);
                for branch in branches {
                    match branch {
                        Value::LyObj(obj) => {
                            assert_eq!(obj.type_name(), "GitBranch");
                        }
                        _ => panic!("Expected GitBranch LyObj"),
                    }
                }
            }
            _ => panic!("Expected list of branches"),
        }
    }
}