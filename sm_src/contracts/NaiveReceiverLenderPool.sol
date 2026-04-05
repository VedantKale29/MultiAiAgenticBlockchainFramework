// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * NaiveReceiverLenderPool.sol
 * ============================
 * STAGE 2 — Flash Loan Vulnerable Pool (DVDF-inspired fixture)
 *
 * PURPOSE:
 *   Provides a vulnerable flash loan pool that the MonitorAgent watches.
 *   When the Attacker contract executes a flash loan, MonitorAgent detects
 *   the FlashLoan event and fires THREAT_DETECTED → triggers the full
 *   Decision → Contract → Governance pipeline.
 *
 * INSPIRED BY:
 *   Damn Vulnerable DeFi (damnvulnerabledefi.xyz, MIT licensed)
 *   — NaiveReceiverLenderPool challenge
 *
 * VULNERABILITY (intentional for testing):
 *   The pool charges a flat fee regardless of the caller — any contract
 *   can drain the receiver's balance by calling flashLoan() repeatedly.
 *   In production DeFi, the receiver would verify msg.sender == pool.
 *
 * HOW IT FITS IN THE PIPELINE:
 *   1. Deploy this contract (scripts/deploy.js)
 *   2. MonitorAgent subscribes to FlashLoan events
 *   3. Run: npx hardhat run scripts/attack.js --network localhost
 *   4. FlashLoan event fires → MonitorAgent detects in <1s
 *   5. THREAT_DETECTED → DecisionAgent → ContractAgent → CircuitBreaker deployed
 *
 * EVENTS EMITTED (watched by MonitorAgent):
 *   FlashLoan(address borrower, uint256 amount, uint256 fee)
 *   Deposit(address from, uint256 amount)
 *   Paused(address by)
 *
 * DEPLOY:
 *   npx hardhat run scripts/deploy.js --network localhost
 *
 * INTERACT (from Python MonitorAgent test):
 *   pool.functions.poolBalance().call()
 *   pool.functions.maxFlashLoan().call()
 */

contract NaiveReceiverLenderPool {

    // ── State ───────────────────────────────────────────────────
    address public owner;
    bool    public paused;           // set by CircuitBreaker response contract
    uint256 public FLASH_LOAN_FEE;  // flat fee in wei per flash loan (0.01 ETH)

    // ── Events ──────────────────────────────────────────────────
    event FlashLoan(
        address indexed borrower,
        uint256         amount,
        uint256         fee
    );
    event Deposit(
        address indexed from,
        uint256         amount
    );
    event Paused(address indexed by);
    event Unpaused(address indexed by);

    // ── Constructor ─────────────────────────────────────────────
    constructor() payable {
        owner           = msg.sender;
        FLASH_LOAN_FEE  = 0.01 ether;
        paused          = false;
    }

    // ── Modifiers ───────────────────────────────────────────────
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    modifier notPaused() {
        require(!paused, "Pool is paused");
        _;
    }

    // ── Deposit ─────────────────────────────────────────────────
    receive() external payable {
        emit Deposit(msg.sender, msg.value);
    }

    function deposit() external payable {
        require(msg.value > 0, "Must deposit ETH");
        emit Deposit(msg.sender, msg.value);
    }

    // ── Flash loan ──────────────────────────────────────────────
    /**
     * Execute a flash loan.
     * VULNERABILITY: any external contract can call this on behalf of
     * any receiver — the receiver's ETH will be drained by the fee.
     */
    function flashLoan(
        address receiver,
        uint256 amount
    ) external notPaused {
        uint256 balanceBefore = address(this).balance;
        require(balanceBefore >= amount, "Not enough pool liquidity");

        // Send ETH to receiver
        (bool sent, ) = receiver.call{value: amount}(
            abi.encodeWithSignature("receiveETH(uint256)", FLASH_LOAN_FEE)
        );
        require(sent, "Flash loan send failed");

        // Receiver must repay amount + fee
        require(
            address(this).balance >= balanceBefore + FLASH_LOAN_FEE,
            "Flash loan not repaid with fee"
        );

        emit FlashLoan(receiver, amount, FLASH_LOAN_FEE);
    }

    // ── Circuit breaker — called by autonomously deployed response contract ──
    function pause() external {
        // Allow owner or any deployed response contract to pause
        paused = true;
        emit Paused(msg.sender);
    }

    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(msg.sender);
    }

    // ── View functions ──────────────────────────────────────────
    function poolBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function maxFlashLoan() external view returns (uint256) {
        return address(this).balance;
    }

    function isPaused() external view returns (bool) {
        return paused;
    }
}
