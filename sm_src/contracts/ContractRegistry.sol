// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * ContractRegistry.sol
 * =====================
 * Immutable on-chain registry of all autonomously deployed response contracts.
 *
 * Every time ContractAgent deploys a CircuitBreaker, AddressBlocklist, or
 * RateLimiter, it calls register() here to create a tamper-proof record.
 *
 * This satisfies the framework's requirement for an on-chain audit trail:
 *   "Smart contract registry -- stores addresses of all autonomously
 *   deployed contracts." (Layer 5, Section 2)
 *
 * DEPLOY:
 *   npx hardhat run scripts/deploy.js --network localhost
 *
 * ABI used by ContractAgent (Python):
 *   register(incidentId bytes32, contractAddress address, templateKey string)
 */

contract ContractRegistry {

    address public owner;

    struct DeploymentRecord {
        bytes32 incidentId;
        address contractAddress;
        string  templateKey;
        address deployedBy;
        uint256 blockNumber;
        uint256 timestamp;
    }

    // incidentId → deployment record
    mapping(bytes32 => DeploymentRecord) private _records;

    // ordered list for enumeration
    bytes32[] private _incidentIds;

    event ContractRegistered(
        bytes32 indexed incidentId,
        address indexed contractAddress,
        string          templateKey,
        uint256         blockNumber
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "ContractRegistry: not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    /**
     * Register a newly deployed response contract.
     * Called by ContractAgent after every deployment.
     * The agentAddress is set as authorised caller in practice,
     * but for the Hardhat demo we allow any caller.
     */
    function register(
        bytes32        incidentId,
        address        contractAddress,
        string calldata templateKey
    ) external {
        require(contractAddress != address(0), "ContractRegistry: zero address");
        require(
            _records[incidentId].blockNumber == 0,
            "ContractRegistry: incident already registered"
        );

        _records[incidentId] = DeploymentRecord({
            incidentId:      incidentId,
            contractAddress: contractAddress,
            templateKey:     templateKey,
            deployedBy:      msg.sender,
            blockNumber:     block.number,
            timestamp:       block.timestamp
        });
        _incidentIds.push(incidentId);

        emit ContractRegistered(
            incidentId,
            contractAddress,
            templateKey,
            block.number
        );
    }

    /**
     * Look up a deployment record by incident ID.
     */
    function getRecord(bytes32 incidentId)
        external
        view
        returns (
            address contractAddress,
            string memory templateKey,
            address deployedBy,
            uint256 blockNumber,
            uint256 timestamp
        )
    {
        DeploymentRecord storage r = _records[incidentId];
        require(r.blockNumber > 0, "ContractRegistry: not found");
        return (
            r.contractAddress,
            r.templateKey,
            r.deployedBy,
            r.blockNumber,
            r.timestamp
        );
    }

    /**
     * Total number of registered deployments.
     */
    function getCount() external view returns (uint256) {
        return _incidentIds.length;
    }

    /**
     * Get incident ID by index (for enumeration).
     */
    function getIncidentId(uint256 index) external view returns (bytes32) {
        require(index < _incidentIds.length, "ContractRegistry: out of range");
        return _incidentIds[index];
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "ContractRegistry: zero address");
        owner = newOwner;
    }
}
