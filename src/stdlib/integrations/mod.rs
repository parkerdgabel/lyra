//! Advanced System Integrations Module
//!
//! This module provides comprehensive integration capabilities for:
//! - Version control systems (Git, GitHub, GitLab)
//! - CI/CD pipeline automation and management
//! - Container orchestration (Docker, Kubernetes)
//! - Cost optimization and resource management
//!
//! All integrations are implemented as Foreign objects following the established pattern
//! of keeping complex state outside the VM core while maintaining thread safety.

pub mod version_control;
pub mod cicd;
pub mod optimization;

use crate::vm::{Value, VmResult};
use crate::stdlib::StandardLibrary;

impl StandardLibrary {
    /// Register all integration functions
    pub fn register_integration_functions(&mut self) {
        // Version Control Integration Functions (10 functions)
        self.register("GitRepository", version_control::git_repository);
        self.register("GitCommit", version_control::git_commit);
        self.register("GitBranch", version_control::git_branch);
        self.register("GitMerge", version_control::git_merge);
        self.register("GitPull", version_control::git_pull);
        self.register("GitPush", version_control::git_push);
        self.register("CommitAnalysis", version_control::commit_analysis);
        self.register("BranchStrategy", version_control::branch_strategy);
        self.register("GitHooks", version_control::git_hooks);
        self.register("CodeReview", version_control::code_review);

        // CI/CD Pipeline Functions (10 functions)
        self.register("PipelineCreate", cicd::pipeline_create);
        self.register("BuildStage", cicd::build_stage);
        self.register("TestStage", cicd::test_stage);
        self.register("DeployStage", cicd::deploy_stage);
        self.register("PipelineTrigger", cicd::pipeline_trigger);
        self.register("ArtifactManagement", cicd::artifact_management);
        self.register("EnvironmentPromotion", cicd::environment_promotion);
        self.register("PipelineMonitoring", cicd::pipeline_monitoring);
        self.register("QualityGates", cicd::quality_gates);
        self.register("DeploymentRollback", cicd::deployment_rollback);

        // Cost Optimization Functions (5 functions)
        self.register("ResourceUsage", optimization::resource_usage);
        self.register("CostAnalysis", optimization::cost_analysis);
        self.register("RightSizing", optimization::right_sizing);
        self.register("CostAlerts", optimization::cost_alerts);
        self.register("BudgetManagement", optimization::budget_management);
    }
}