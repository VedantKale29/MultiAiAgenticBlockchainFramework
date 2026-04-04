// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * GovernanceContract.sol
 * ======================
 * On-chain governance for the fraud detection agent system.
 *
 * PATTERN: Propose → Timelock → Execute  (or Cancel during window)
 *
 * HOW IT WORKS:
 *   1. GovernanceAgent calls propose(param, newValue, reason)
 *      -- any address can propose (no privileged key required)
 *   2. The proposal is queued with a time-lock delay (default 60s on Hardhat)
 *   3. During the time-lock window, the owner can call cancel(proposalId)
 *   4. After the delay, anyone calls execute(proposalId) and the
 *      parameter is updated automatically
 *
 * MAPS TO ROADMAP:
 *   Section 4.4 (revised): "AI proposes; TimelockController executes
 *   after grace period; human cancel available during window."
 *   Satisfies RO5: fully autonomous on-chain self-governance.
 *
 * GOVERNED PARAMETERS:
 *   "tau_alert"   -- fraud alert threshold (scaled ×1e18)
 *   "tau_block"   -- auto-block threshold  (scaled ×1e18)
 *   "w"           -- RF/IF fusion weight    (scaled ×1e18)
 *   "escalation_n"-- repeat-alert count before block (integer, not scaled)
 *
 * DEPLOY:
 *   npx hardhat run scripts/deploy.js --network localhost
 *
 * READING PARAMETERS from Python (Web3.py):
 *   contract.functions.getParameter("tau_alert").call()
 *   → returns uint256 scaled by 1e18
 *   → divide by 1e18 to get float: 487000000000000000 / 1e18 = 0.487
 */

contract GovernanceContract {

    // ── State ───────────────────────────────────────────────────
    address public owner;
    uint256 public timelockDelay;   // seconds

    // Current parameter values (scaled ×1e18 for float precision)
    mapping(string => uint256) private _params;

    // ── Proposal ────────────────────────────────────────────────
    enum ProposalStatus { Pending, Executed, Cancelled }

    struct Proposal {
        string          param;
        uint256         newValue;
        string          reason;
        address         proposer;
        uint256         eta;        // earliest execution timestamp
        ProposalStatus  status;
    }

    mapping(bytes32 => Proposal) public proposals;
    bytes32[] public proposalIds;

    // ── Events ──────────────────────────────────────────────────
    event ProposalCreated(
        bytes32 indexed proposalId,
        string  param,
        uint256 oldValue,
        uint256 newValue,
        uint256 eta,
        string  reason
    );
    event ProposalExecuted(
        bytes32 indexed proposalId,
        string  param,
        uint256 newValue
    );
    event ProposalCancelled(
        bytes32 indexed proposalId
    );
    event ParameterUpdated(
        string  indexed param,
        uint256 oldValue,
        uint256 newValue,
        uint256 blockNumber
    );

    // ── Modifiers ───────────────────────────────────────────────
    modifier onlyOwner() {
        require(msg.sender == owner, "GovernanceContract: not owner");
        _;
    }

    // ── Constructor ─────────────────────────────────────────────
    constructor(uint256 _timelockDelay) {
        owner         = msg.sender;
        timelockDelay = _timelockDelay;

        // Default parameter values (all scaled ×1e18)
        _params["tau_alert"]    = 487000000000000000;  // 0.487
        _params["tau_block"]    = 587000000000000000;  // 0.587
        _params["w"]            = 700000000000000000;  // 0.700
        _params["escalation_n"] = 3000000000000000000; // 3 (integer stored scaled)
    }

    // ── Propose ─────────────────────────────────────────────────
    /**
     * Submit a governance proposal. Any address may call this.
     * Returns the proposalId (bytes32 hash).
     *
     * Called by GovernanceAgent.propose() in Python.
     */
    function propose(
        string calldata param,
        uint256         newValue,
        string calldata reason
    ) external returns (bytes32 proposalId) {
        require(bytes(param).length > 0,  "GovernanceContract: empty param");
        require(newValue > 0,             "GovernanceContract: zero value");

        uint256 eta = block.timestamp + timelockDelay;

        // Deterministic proposal ID from content + timestamp
        proposalId = keccak256(
            abi.encodePacked(param, newValue, reason, block.timestamp, msg.sender)
        );
        require(
            proposals[proposalId].eta == 0,
            "GovernanceContract: proposal already exists"
        );

        proposals[proposalId] = Proposal({
            param:     param,
            newValue:  newValue,
            reason:    reason,
            proposer:  msg.sender,
            eta:       eta,
            status:    ProposalStatus.Pending
        });
        proposalIds.push(proposalId);

        emit ProposalCreated(
            proposalId,
            param,
            _params[param],
            newValue,
            eta,
            reason
        );
    }

    // ── Execute ─────────────────────────────────────────────────
    /**
     * Execute a proposal after the timelock delay has elapsed.
     * Anyone may call this once the eta has passed.
     */
    function execute(bytes32 proposalId) external {
        Proposal storage p = proposals[proposalId];
        require(p.eta > 0,                          "GovernanceContract: unknown proposal");
        require(p.status == ProposalStatus.Pending,  "GovernanceContract: not pending");
        require(block.timestamp >= p.eta,            "GovernanceContract: timelock active");

        uint256 oldValue = _params[p.param];
        _params[p.param] = p.newValue;
        p.status = ProposalStatus.Executed;

        emit ProposalExecuted(proposalId, p.param, p.newValue);
        emit ParameterUpdated(p.param, oldValue, p.newValue, block.number);
    }

    // ── Cancel ──────────────────────────────────────────────────
    /**
     * Cancel a pending proposal. Only the owner (human override) can cancel.
     * This is the safety mechanism during the timelock grace period.
     */
    function cancel(bytes32 proposalId) external onlyOwner {
        Proposal storage p = proposals[proposalId];
        require(p.eta > 0,                         "GovernanceContract: unknown proposal");
        require(p.status == ProposalStatus.Pending, "GovernanceContract: not pending");

        p.status = ProposalStatus.Cancelled;
        emit ProposalCancelled(proposalId);
    }

    // ── Read ────────────────────────────────────────────────────
    /**
     * Get the current value of a parameter (scaled ×1e18).
     * Divide by 1e18 in Python to get the float value.
     */
    function getParameter(string calldata param)
        external
        view
        returns (uint256)
    {
        return _params[param];
    }

    /**
     * Get all current parameters at once for the agent's startup cache.
     */
    function getAllParameters()
        external
        view
        returns (
            uint256 tau_alert,
            uint256 tau_block,
            uint256 w,
            uint256 escalation_n
        )
    {
        return (
            _params["tau_alert"],
            _params["tau_block"],
            _params["w"],
            _params["escalation_n"]
        );
    }

    /**
     * Get proposal details by ID.
     */
    function getProposal(bytes32 proposalId)
        external
        view
        returns (
            string memory param,
            uint256 newValue,
            string memory reason,
            address proposer,
            uint256 eta,
            uint8   status
        )
    {
        Proposal storage p = proposals[proposalId];
        return (
            p.param,
            p.newValue,
            p.reason,
            p.proposer,
            p.eta,
            uint8(p.status)
        );
    }

    /**
     * Get all proposal IDs (for audit enumeration).
     */
    function getProposalCount() external view returns (uint256) {
        return proposalIds.length;
    }

    // ── Admin ───────────────────────────────────────────────────
    function setTimelockDelay(uint256 newDelay) external onlyOwner {
        timelockDelay = newDelay;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "GovernanceContract: zero address");
        owner = newOwner;
    }
}
